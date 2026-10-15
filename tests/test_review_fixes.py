"""Precision provenance, MoE update semantics and representative validation."""
from copy import deepcopy
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import ConcatDataset, Dataset
from safetensors.torch import load_file, save_file

from lrvla.inference_profile import apply_inference_profile
from lrvla.training_checkpoint import overlay_base_state_items, restore_trainable_base
from lrvla.training_data import MODEL_KEYS
from lrvla.training_distributed_checkpoint import FORMAT, load_distributed_export
from lrvla.training_identity import base_checkpoint_identity
from lrvla.training_instructions import load_seen_pool, sample_seen, sampling_identity
from lrvla.training_models import LoRALinear, moe_auxiliary_losses
from lrvla.training_validation import validate_stratified, validation_draws, validation_identity, validation_samples

ROOT = Path(__file__).resolve().parents[1]


def test_original_fp32_frozen_and_expert_master_are_preserved(tmp_path):
    frozen = torch.tensor([1.0001, 2.0002])
    save_file({"frozen": frozen, "expert": torch.tensor([3.0003])}, str(tmp_path / "model.safetensors"))
    master = torch.tensor([4.0004])
    selection = {"trainable_names": ["expert"], "lora_modules": []}
    result = dict(overlay_base_state_items(tmp_path, {"expert": master}, selection,
                                         expected_base_identity=base_checkpoint_identity(tmp_path)))
    assert not torch.equal(frozen, frozen.bfloat16().float())
    assert torch.equal(result["frozen"], frozen)
    assert torch.equal(result["expert"], master)
    with pytest.raises(ValueError, match="base_identity"):
        overlay_base_state_items(tmp_path, {"expert": master}, selection, expected_base_identity={})
    with pytest.raises(ValueError, match="FP32 master"):
        overlay_base_state_items(tmp_path, {"expert": master.bfloat16()}, selection,
                                 expected_base_identity=base_checkpoint_identity(tmp_path))


def test_lora_merges_against_original_base_not_rounded_training_storage(tmp_path):
    base = torch.full((3, 2), 1.0001)
    save_file({"layer.weight": base}, str(tmp_path / "model.safetensors"))
    a, b = torch.tensor([[0.001, 0.003]]), torch.tensor([[0.1], [0.2], [0.3]])
    selection = {"trainable_names": ["layer.lora_A", "layer.lora_B"], "lora_modules": ["layer"],
                 "lora": {"rank": 1, "alpha": 2}}
    result = dict(overlay_base_state_items(tmp_path, {"layer.lora_A": a, "layer.lora_B": b}, selection,
                                         expected_base_identity=base_checkpoint_identity(tmp_path)))
    assert set(result) == {"layer.weight"}
    assert torch.equal(result["layer.weight"], base + b @ a * 2)
    bad = {"layer.lora_A": torch.zeros(1, 3), "layer.lora_B": b}
    with pytest.raises(ValueError, match="shape mismatch"):
        overlay_base_state_items(tmp_path, bad, selection, expected_base_identity=base_checkpoint_identity(tmp_path))


def test_new_training_masters_start_from_original_fp32(tmp_path):
    model = nn.Linear(2, 1)
    with torch.no_grad():
        model.weight.fill_(1.0001)
        model.bias.fill_(0.0001234)
    original = {key: value.clone() for key, value in model.state_dict().items()}
    save_file(original, str(tmp_path / "model.safetensors"))
    model.bfloat16().float()
    assert not torch.equal(model.weight, original["weight"])
    restore_trainable_base(model, tmp_path)
    for key, value in model.state_dict().items():
        assert torch.equal(value, original[key])


def test_dcp_export_reads_model_only_without_training_world_size(tmp_path, monkeypatch):
    import torch.distributed.checkpoint as dcp
    checkpoint = tmp_path / "distributed"
    checkpoint.mkdir()
    master = torch.full((2, 3), 1.0001)
    dcp.save({"model": {"policy.expert": master}, "optimizer": {"huge_state": torch.ones(20)}},
             checkpoint_id=checkpoint / "shards")
    run = {"format": FORMAT, "world_size": 4, "metadata": {"selection": {"trainable_names": ["expert"]}},
           "state": {"global_step": 1000}}
    (checkpoint / "run_state.json").write_text(json.dumps(run))
    (checkpoint / "complete.json").write_text(json.dumps({"format": FORMAT, "world_size": 4,
        "run_state_sha256": hashlib.sha256((checkpoint / "run_state.json").read_bytes()).hexdigest()}))
    original_read = dcp.FileSystemReader.read_data
    requested = []
    def read(self, plan, planner):
        requested.extend(item.storage_index.fqn for item in plan.items)
        return original_read(self, plan, planner)
    monkeypatch.setattr(dcp.FileSystemReader, "read_data", read)
    result, metadata = load_distributed_export(checkpoint)
    assert torch.equal(result["expert"], master)
    assert metadata["world_size"] == 4
    assert requested and set(requested) == {"model.policy.expert"}
    (checkpoint / "run_state.json").write_text("{}")
    with pytest.raises(ValueError, match="completion marker"):
        load_distributed_export(checkpoint)


def test_standalone_export_preserves_old_export_and_checks_run_assets(tmp_path):
    import yaml
    spec = importlib.util.spec_from_file_location("export_saved_cli", ROOT / "scripts/export_checkpoint.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    run = tmp_path / "run"
    step = run / "checkpoints/global_step_10"
    assets = run / "assets/qwen3_vl"
    step.mkdir(parents=True)
    assets.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        (assets / name).write_text("{}")
    (run / "assets/norm_stats.json").write_text("{}")
    base = tmp_path / "base"
    base.mkdir()
    frozen, master = torch.tensor([1.0001]), torch.tensor([2.0002])
    save_file({"frozen": frozen, "expert": torch.tensor([0.0001])}, str(base / "model.safetensors"))
    geometry = {"chunk_size": 50, "max_action_dim": 2, "max_state_dim": 2, "img_size": 256}
    selection = {"trainable_names": ["expert"], "lora_modules": []}
    payload = {"trainable_state": {"expert": master}, "state": {"global_step": 10}, "metadata": {
        "base_identity": base_checkpoint_identity(base), "selection": selection, "model_geometry": geometry,
        "qwen_config_sha256": hashlib.sha256(b"{}").hexdigest(), "norm_sha256": hashlib.sha256(b"{}").hexdigest()}}
    checkpoint = step / "training.pt"
    torch.save(payload, checkpoint)
    config = {"model": {key: geometry[key] for key in ("chunk_size", "max_action_dim", "max_state_dim")},
              "train": {}, "data": {"img_size": 256}}
    (run / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(config))
    old = step / "hf_ckpt"
    old.mkdir()
    (old / "untouched").write_text("old bf16 export")
    output = cli.export_saved(checkpoint, base)
    weights = load_file(output / "model-00001.safetensors")
    assert torch.equal(weights["frozen"], frozen) and torch.equal(weights["expert"], master)
    assert (old / "untouched").read_text() == "old bf16 export"
    assert json.loads((output / "config.json").read_text())["torch_dtype"] == "float32"
    config["model"]["chunk_size"] = 10
    (run / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(config))
    bad_output = step / "hf_ckpt_bad"
    with pytest.raises(ValueError, match="model_geometry"):
        cli.export_saved(checkpoint, base, bad_output)
    assert not bad_output.exists()


def test_moe_values_and_gradients_match_upstream_cpu_reference():
    spec = importlib.util.spec_from_file_location("upstream_moe_reference", ROOT / "vendor/lingbot-vla-v2/lingbotvla/ops/moe_loss.py")
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    torch.manual_seed(17)
    logits = [torch.randn(3 * 7, 8, requires_grad=True) for _ in range(2)]
    balance, z, loads = moe_auxiliary_losses(logits, 3, top_k=4)
    expected = torch.stack(reference.sequence_wise_balance_loss(tuple(logits), 4,
                           seq_lengths=[7] * 3, score_func="sigmoid")).mean()
    expected_z = torch.stack([x.logsumexp(-1).square().mean() for x in logits]).mean()
    torch.testing.assert_close(balance.mean(), expected)
    torch.testing.assert_close(z.mean(), expected_z)
    actual_grad = torch.autograd.grad(balance.mean() * .001 + z.mean() * .0001, logits, retain_graph=True)
    ref_grad = torch.autograd.grad(expected * .001 + expected_z * .0001, logits)
    for actual, expected in zip(actual_grad, ref_grad):
        torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-10)
    assert set(loads) == {"layer_0", "layer_1"}
    assert loads["layer_0"]["tokens_per_expert"].sum() == 3 * 7 * 4


@pytest.mark.parametrize("world,sizes", [(1, [2, 3]), (2, [1, 1, 1]), (4, [2])])
def test_moe_plus_bc_uneven_tail_gradients_equal_large_batch(world, sizes):
    torch.manual_seed(24)
    x = torch.randn(8, 5, 3)
    mask = (torch.arange(5)[None, :, None] < (torch.arange(8) % 5 + 1)[:, None, None]).expand(-1, -1, 8)
    reference = nn.Linear(3, 8)
    count, samples = float(mask.sum()), len(x)
    def numerator(model, ids):
        out = model(x[ids])
        balance, z, _ = moe_auxiliary_losses([out.flatten(0, 1)], len(ids))
        return (out.square() * mask[ids]).sum() + (count / samples) * (.001 * balance.sum() + .0001 * z.sum())
    (numerator(reference, list(range(samples))) / count).backward()
    rank_models = [deepcopy(reference) for _ in range(world)]
    for rank, model in enumerate(rank_models):
        model.zero_grad()
        ids, offset, chunk = list(range(rank, samples, world)), 0, 0
        while offset < len(ids):
            width = sizes[chunk % len(sizes)]
            numerator(model, ids[offset:offset + width]).backward()
            offset += width
            chunk += 1
    for i, expected in enumerate(reference.parameters()):
        averaged = sum(list(model.parameters())[i].grad for model in rank_models) / world
        actual = averaged / (count / world)
        torch.testing.assert_close(actual, expected.grad, rtol=1e-5, atol=1e-6)


def test_moe_gradient_reaches_lora_through_frozen_router():
    adapter = LoRALinear(nn.Linear(3, 3), rank=2)
    router = nn.Linear(3, 8).requires_grad_(False)
    balance, z, _ = moe_auxiliary_losses([router(adapter(torch.randn(10, 3)))], 2)
    (balance.sum() + z.sum()).backward()
    assert adapter.lora_B.grad.abs().sum() > 0
    assert router.weight.grad is None and adapter.base.weight.grad is None


class EpisodeDataset(Dataset):
    def __init__(self, task, episodes):
        self.task = task
        self.dataset = SimpleNamespace(hf_dataset={"episode_index": episodes})
        self.sample_indices = None
    def __len__(self):
        return len(self.dataset.hf_dataset["episode_index"])
    def __getitem__(self, index):
        result = {key: torch.zeros(1) for key in MODEL_KEYS}
        result.update(actions=torch.zeros(3, 2), joint_mask=torch.ones(3, 2, dtype=torch.bool),
                      lang_tokens=torch.ones(index % 3 + 1, dtype=torch.long),
                      lang_masks=torch.ones(index % 3 + 1, dtype=torch.bool))
        return result


def test_validation_covers_every_task_episode_without_loading_rgb():
    dataset = ConcatDataset([EpisodeDataset(f"task{i}", [10, 10, 20, 30, 30, 40, 50]) for i in range(50)])
    selected = validation_samples(dataset)
    assert selected == validation_samples(dataset)
    assert len(selected) == 250
    coverage = validation_identity(selected)
    assert coverage["tasks"] == 50 and coverage["episodes"] == 250
    assert {dataset[row["index"]] is not None for row in selected} == {True}
    for row in selected:
        child = dataset.datasets[row["index"] // 7]
        assert child.task == row["task"]
        assert child.dataset.hf_dataset["episode_index"][row["index"] % 7] == row["episode"]
    assert validation_samples(ConcatDataset([EpisodeDataset("empty", [])])) == []


def test_validation_noise_and_aggregate_independent_of_batch_and_world(monkeypatch):
    dataset = ConcatDataset([EpisodeDataset("a", [2, 2, 4]), EpisodeDataset("b", [9, 12, 15])])
    selected = validation_samples(dataset)
    def loss_fn(model, batch, *, noise, time, **kwargs):
        # Noise, time and original mask are really consumed by validation.
        values = noise.square().flatten(1).sum(1) + time
        return values.sum(), {"per_sample_bc_sum": values, "per_sample_valid_coordinates": batch["joint_mask"].sum((1, 2))}
    model = nn.Linear(1, 1)
    saved = torch.get_rng_state().clone()
    reference = validate_stratified(model, dataset, "cpu", batch_size=2, samples=selected, loss_fn=loss_fn)
    for world in (1, 2, 4):
        totals = []
        monkeypatch.setattr(torch.distributed, "all_reduce", lambda value: totals.append(value.clone()))
        for rank in range(world):
            validate_stratified(model, dataset, "cpu", batch_size=2, samples=selected,
                                loss_fn=loss_fn, rank=rank, world_size=world)
        if world > 1:
            combined = torch.stack(totals).sum(0).sum(0)
            assert float(combined[0] / combined[1]) == pytest.approx(reference["validation_bc_loss"])
        ids = [row["index"] for row in selected]
        all_noise, all_time = validation_draws(ids, (3, 2))
        for rank in range(world):
            noise, time = validation_draws(ids[rank::world], (3, 2))
            assert torch.equal(noise, all_noise[rank::world]) and torch.equal(time, all_time[rank::world])
    assert torch.equal(saved, torch.get_rng_state()) and model.training


def test_validation_uses_coordinate_mean_for_uneven_episode_tails():
    class Uneven(EpisodeDataset):
        def __getitem__(self, index):
            result = super().__getitem__(index)
            result["actions"].fill_(2 if index == 0 else 6)
            if index == 0:
                result["joint_mask"][1:] = False
            return result
    dataset = ConcatDataset([Uneven("a", [2, 4])])
    def loss_fn(model, batch, **kwargs):
        counts = batch["joint_mask"].sum((1, 2))
        return torch.tensor(0.), {"per_sample_bc_sum": batch["actions"][:, 0, 0] * counts,
                                  "per_sample_valid_coordinates": counts}
    result = validate_stratified(nn.Linear(1, 1), dataset, "cpu", batch_size=2,
                                 samples=validation_samples(dataset), loss_fn=loss_fn)
    assert result["validation_bc_loss"] == pytest.approx((2 * 2 + 6 * 6) / 8)


def test_seen_pool_stays_episode_local_and_identity_survives_relocation(tmp_path):
    pool = {"schema": "clean_episode_seen_v1", "clean_only": True, "tasks": {"a": {
        "source_sha256": "raw", "episodes": {"0": ["zero a", "zero b"], "1": ["one"]}}}}
    manifest = {"datasets": [{"task": "a", "episode_ids": [0, 1], "source": {"sha256": "raw"}}]}
    path = tmp_path / "pool.json"
    path.write_text(json.dumps(pool))
    loaded, digest = load_seen_pool(path, manifest)
    choices = [sample_seen(loaded, "a", 0, frame, seed=42, epoch=2) for frame in range(20)]
    assert set(choices) == {"zero a", "zero b"}
    assert choices == [sample_seen(loaded, "a", 0, frame, seed=42, epoch=2) for frame in range(20)]
    assert sample_seen(loaded, "a", 1, 2, seed=42, epoch=2) == "one"
    assert sampling_identity({"image_augment": False}, digest) != sampling_identity({"image_augment": True}, digest)
    moved = tmp_path / "moved.json"
    moved.write_text(json.dumps(pool, indent=4))
    assert load_seen_pool(moved, manifest)[1] == digest


def test_effective_inference_steps_reach_sampling_model():
    policy = SimpleNamespace(vla=SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(chunk_size=50))),
                             use_length=25, use_compile=False)
    apply_inference_profile(policy, 20)
    assert policy.vla.model.config.num_steps == 20
    with pytest.raises(ValueError, match="positive"):
        apply_inference_profile(policy, 0)


def test_eval_yaml_defaults_overrides_and_profile_resume_identity(tmp_path):
    spec = importlib.util.spec_from_file_location("evaluate_profile", ROOT / "scripts/evaluate.py")
    cli = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(cli)
    args = cli.parse_args(["--checkpoint", "/model", "--backend", "streamed", "--num-steps", "20"])
    assert (args.backend, args.precision, args.use_length, args.num_steps, args.compile) == ("streamed", "fp32", 50, 20, False)
    plan = {"num_steps": 10, "compile": False, "action_smoothing": "none"}
    (tmp_path / "evaluation_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match="num_steps"):
        cli.prepare_resumed_plan(tmp_path, {**plan, "num_steps": 20})
    path = tmp_path / "eval.yaml"
    path.write_text("num_steps: 10.5\n")
    with pytest.raises(ValueError, match="integer"):
        cli.parse_args(["--config", str(path), "--checkpoint", "/model"])
