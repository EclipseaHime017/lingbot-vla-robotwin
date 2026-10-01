from pathlib import Path
import sys
import json
import shutil
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn

from lrvla.training_data import EpochBatchSampler, LocalCleanDataset, MODEL_KEYS, episode_split, epoch_dataloader
from lrvla.training_models import LoRALinear, configure_trainable, fused_storage_torch_forward, masked_bc_loss, sdpa_attention
from lrvla.training_smoke import TinyPolicy, run_smoke
from lrvla.training_checkpoint import resume_training_checkpoint, save_training_checkpoint
from lrvla.training_identity import base_checkpoint_identity, manifest_content_sha256


def test_mask_excludes_padding_and_backpropagates_only_valid_actions():
    prediction = torch.tensor([[[2.0, 1000.0], [4.0, -999.0]]], requires_grad=True)
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[[True, False], [False, False]]])
    loss = masked_bc_loss(prediction, target, mask, "fm")
    assert loss.item() == 4
    loss.backward()
    torch.testing.assert_close(prediction.grad, torch.tensor([[[4.0, 0], [0, 0]]]))
    with pytest.raises(ValueError, match="no valid"):
        masked_bc_loss(prediction, target, torch.zeros_like(mask))


def test_final_episode_frame_ignores_repeated_padding_targets():
    # A final real action is followed by three repeated end-of-episode targets.
    # Simulate the actual LeRobot/FeatureTransform boundary without video/model assets.
    raw = {"observation.state": torch.zeros(14), "action": torch.zeros(4, 14)}
    class Backend:
        def __getitem__(self, idx):
            assert idx == 0
            return raw
    transformed = {key: torch.empty(0) for key in MODEL_KEYS}
    transformed.update(actions=torch.zeros(4, 2), joint_mask=torch.ones(4, 2, dtype=torch.bool),
                       action_is_pad=torch.tensor([False, True, True, True]))
    dataset = LocalCleanDataset.__new__(LocalCleanDataset)
    dataset.dataset = Backend()
    dataset.transform = SimpleNamespace(apply=lambda _: dict(transformed))
    item = dataset[0]
    assert item["joint_mask"][0].all() and not item["joint_mask"][1:].any()
    prediction = torch.tensor([[[1.0, 1.0], [1000.0, 1000.0], [1000.0, 1000.0], [1000.0, 1000.0]]], requires_grad=True)
    loss = masked_bc_loss(prediction, item["actions"][None], item["joint_mask"][None], "fm")
    assert loss.item() == 1.0
    loss.backward()
    assert not prediction.grad[:, 1:].any()


def test_lora_zero_initialization_and_merge_equivalence():
    torch.manual_seed(2)
    base = nn.Linear(5, 3)
    adapter = LoRALinear(base, rank=2, alpha=4)
    x = torch.randn(4, 5)
    torch.testing.assert_close(adapter(x), base(x), rtol=0, atol=0)
    with torch.no_grad():
        adapter.lora_B.normal_()
    expected = torch.nn.functional.linear(x, adapter.merged_weight(), base.bias)
    torch.testing.assert_close(adapter(x), expected)
    adapter(x).square().mean().backward()
    assert base.weight.grad is None
    assert adapter.lora_A.grad.abs().sum() > 0


def test_resume_accepts_moved_paths_and_rejects_changed_base_or_clean_data(tmp_path):
    from safetensors.torch import save_file
    original, relocated = tmp_path / "laptop-base", tmp_path / "server-base"
    original.mkdir()
    relocated.mkdir()
    save_file({"frozen": torch.arange(8).float()}, str(original / "model.safetensors"))
    shutil.copy2(original / "model.safetensors", relocated / "model.safetensors")
    identity = base_checkpoint_identity(original)
    assert base_checkpoint_identity(relocated) == identity
    manifest = {"training_list": "/old/clean.txt", "datasets": [
        {"task": "adjust_bottle", "path": "/old/clean", "source": {"sha256": "frozen-raw"},
         "episode_ids": list(range(50)), "metadata_sha256": "frozen-metadata"}]}
    moved = json.loads(json.dumps(manifest))
    moved["training_list"], moved["datasets"][0]["path"] = "/server/clean.txt", "/server/clean"
    content = manifest_content_sha256(manifest)
    assert manifest_content_sha256(moved) == content
    model = nn.Linear(2, 1)
    optimizer = torch.optim.AdamW(model.parameters())
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1)
    metadata = {"base_checkpoint": str(original), "base_identity": identity,
                "manifest_sha256": "original-path-bearing-file", "manifest_content_sha256": content}
    checkpoint = tmp_path / "training.pt"
    save_training_checkpoint(model, optimizer, scheduler, checkpoint, {"global_step": 9}, metadata)
    portable = {**metadata, "base_checkpoint": str(relocated), "manifest_sha256": "rewritten-paths-file"}
    assert resume_training_checkpoint(model, optimizer, scheduler, checkpoint, expected_metadata=portable)["global_step"] == 9
    # A modified base at the SAME filename/path is rejected by actual byte identity.
    save_file({"frozen": torch.arange(8).float() + 1}, str(relocated / "model.safetensors"))
    portable["base_identity"] = base_checkpoint_identity(relocated)
    with pytest.raises(ValueError, match="base_identity"):
        resume_training_checkpoint(model, optimizer, scheduler, checkpoint, expected_metadata=portable)
    portable["base_identity"] = identity
    moved["datasets"][0]["source"]["sha256"] = "different-clean-release"
    portable["manifest_content_sha256"] = manifest_content_sha256(moved)
    with pytest.raises(ValueError, match="manifest_content_sha256"):
        resume_training_checkpoint(model, optimizer, scheduler, checkpoint, expected_metadata=portable)


def test_incomplete_checkpoint_fails_before_mutating_model(tmp_path):
    from safetensors.torch import save_file
    from lrvla.training_checkpoint import load_safetensors_streamed
    source, target = nn.Linear(3, 2), nn.Linear(3, 2)
    before = {key: value.clone() for key, value in target.state_dict().items()}
    save_file({"weight": source.weight.detach()}, str(tmp_path / "model.safetensors"))
    with pytest.raises(ValueError, match="Incomplete checkpoint"):
        load_safetensors_streamed(target, tmp_path)
    for key, value in target.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)


def test_partial_expert_freezes_vlm_and_earlier_action_layers():
    model = TinyPolicy()
    result = configure_trainable(model, {"mode": "expert", "expert_last_n_layers": 1})
    assert result["expert_last_n_layers"] == 1
    params = dict(model.named_parameters())
    assert all(not p.requires_grad for name, p in params.items() if "qwenvl." in name or "layers.0." in name)
    assert all(p.requires_grad for name, p in params.items() if "layers.1." in name)


def test_resumable_sampler_and_episode_holdout():
    all_batches = list(EpochBatchSampler(11, 3, 42, epoch=2))
    resumed = list(EpochBatchSampler(11, 3, 42, epoch=2, offset=2))
    assert resumed == all_batches[2:]
    assert sorted(sum(all_batches, [])) == list(range(11))
    train, validation = episode_split(range(50), 5)
    assert train == list(range(45)) and validation == list(range(45, 50))
    assert set(train).isdisjoint(validation)


def test_resumed_data_iterator_preserves_next_batch_and_noise_rng():
    from torch.utils.data import TensorDataset, default_collate
    dataset = TensorDataset(torch.arange(20))
    torch.manual_seed(28)
    full = iter(epoch_dataloader(dataset, 3, 42, epoch=1, collate_fn=default_collate))
    next(full)
    next(full)
    saved_rng = torch.get_rng_state()
    next_batch, noise = next(full)[0], torch.randn(5)
    torch.set_rng_state(saved_rng)
    resumed = iter(epoch_dataloader(dataset, 3, 42, epoch=1, offset=2, collate_fn=default_collate))
    torch.testing.assert_close(next(resumed)[0], next_batch, rtol=0, atol=0)
    torch.testing.assert_close(torch.randn(5), noise, rtol=0, atol=0)


def test_deployment_assets_survive_run_migration_and_unrelated_cwd(tmp_path, monkeypatch):
    from lrvla.training import portable_inference_cli
    from lrvla.training_checkpoint import run_relative_path
    original = tmp_path / "original"
    (original / "assets" / "qwen3_vl").mkdir(parents=True)
    (original / "configs" / "robot_configs").mkdir(parents=True)
    stats = {"norm_stats": {"action.arm.position": {"mean": [0.1, 0.2]}}}
    (original / "assets" / "norm_stats.json").write_text(json.dumps(stats))
    (original / "assets" / "qwen3_vl" / "config.json").write_text('{"model_type":"qwen3_vl"}')
    cli = portable_inference_cli({"model": {"tokenizer_path": "/old/machine/qwen"}, "data": {
        "norm_stats_file": "/old/run/norm.json", "robot_config_root": "/old/run/robots", "train_path": "/old/run/data"}})
    moved = tmp_path / "server" / "run"
    shutil.copytree(original, moved)
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    monkeypatch.chdir(unrelated)
    assert json.loads(run_relative_path(moved, cli["data"]["norm_stats_file"]).read_text()) == stats
    assert run_relative_path(moved, cli["data"]["robot_config_root"]).is_dir()
    assert json.loads((run_relative_path(moved, cli["model"]["tokenizer_path"]) / "config.json").read_text())["model_type"] == "qwen3_vl"


def test_sdpa_matches_dense_reference_with_gqa_and_mask():
    torch.manual_seed(4)
    q, k, v = torch.randn(2, 5, 4, 3), torch.randn(2, 5, 2, 3), torch.randn(2, 5, 2, 3)
    mask = torch.ones(2, 5, 5, dtype=torch.bool).tril()
    actual = sdpa_attention(q, k, v, mask)
    kk, vv = k.repeat_interleave(2, dim=2), v.repeat_interleave(2, dim=2)
    scores = torch.einsum("bqhd,bkhd->bhqk", q, kk) / 3**0.5
    probabilities = scores.masked_fill(~mask[:, None], float("-inf")).softmax(-1)
    expected = torch.einsum("bhqk,bkhd->bqhd", probabilities, vv).flatten(2)
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)


def test_fused_storage_reference_has_input_and_weight_gradients():
    torch.manual_seed(5)
    experts = SimpleNamespace(gate_proj=nn.Parameter(torch.randn(3, 4, 2)),
                              up_proj=nn.Parameter(torch.randn(3, 4, 2)),
                              down_proj=nn.Parameter(torch.randn(3, 2, 4)))
    x = torch.randn(6, 2, requires_grad=True)
    selected = torch.tensor([[0, 1], [1, 2], [2, 0], [0, 2], [1, 0], [2, 1]])
    route = torch.rand(6, 2, requires_grad=True)
    result = fused_storage_torch_forward(experts, None, 3, route, selected, x)
    reference = []
    for i in range(6):
        terms = []
        for j, expert in enumerate(selected[i]):
            h = torch.nn.functional.silu(experts.gate_proj[expert] @ x[i]) * (experts.up_proj[expert] @ x[i])
            terms.append((experts.down_proj[expert] @ h) * route[i, j])
        reference.append(sum(terms))
    torch.testing.assert_close(result, torch.stack(reference))
    result.square().mean().backward()
    assert all(t.grad is not None and torch.isfinite(t.grad).all()
               for t in (x, route, experts.gate_proj, experts.up_proj, experts.down_proj))


def test_official_coupled_forward_frozen_prefix_cache_matches_joint_gradients():
    """Use the actual upstream coupled loop, with tiny two-stage decoder modules."""
    from lrvla.training_runtime import add_vendor_path
    add_vendor_path(Path(__file__).resolve().parents[1])
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import QwenvlWithExpertV2Model

    class Decoder(nn.Module):
        def __init__(self, action):
            super().__init__()
            self.action = action
            self.q, self.k, self.v, self.o = nn.Linear(8, 12), nn.Linear(8, 6), nn.Linear(8, 6), nn.Linear(12, 8)

        def forward(self, hidden, attention=None, start=None, end=None, compute_kqv=False, output_atten=False, **kwargs):
            if compute_kqv:
                b, s, _ = hidden.shape
                return self.q(hidden).view(b, s, 4, 3), self.k(hidden).view(b, s, 2, 3), self.v(hidden).view(b, s, 2, 3)
            assert output_atten
            output = hidden + torch.tanh(self.o(attention[:, start:end]))
            return (output, None) if self.action else output

    torch.manual_seed(41)
    core = QwenvlWithExpertV2Model.__new__(QwenvlWithExpertV2Model)
    nn.Module.__init__(core)
    core.config = SimpleNamespace(attention_implementation="eager", final_norm_adanorm=False,
                                  qwen_expert_config=SimpleNamespace(num_hidden_layers=2))
    core.qwenvl = nn.Module()
    core.qwenvl.config = SimpleNamespace(text_config=SimpleNamespace(num_hidden_layers=2, num_attention_heads=4))
    core.qwenvl.model = nn.Module()
    core.qwenvl.model.language_model = nn.Module()
    core.qwenvl.model.language_model.layers = nn.ModuleList([Decoder(False), Decoder(False)])
    core.qwenvl.model.language_model.norm = nn.LayerNorm(8)
    core.qwenvl.requires_grad_(False)
    core.qwen_expert = nn.Module()
    core.qwen_expert.model = nn.Module()
    core.qwen_expert.model.layers = nn.ModuleList([Decoder(True), Decoder(True)])
    core.qwen_expert.model.norm = nn.LayerNorm(8)
    core.apply_mrope = lambda q, k, position: (q, k)
    core.attention_interface = sdpa_attention
    prefix = torch.randn(1, 4, 8)
    suffix = torch.randn(1, 3, 8, requires_grad=True)
    mask = torch.zeros(1, 7, 7, dtype=torch.bool)
    mask[:, :4, :4] = torch.ones(4, 4, dtype=torch.bool).tril()
    mask[:, 4:, :4] = True
    mask[:, 4, 4] = True
    mask[:, 5:, 4:] = True
    positions = torch.zeros(3, 1, 7, dtype=torch.long)
    joint, _, _ = core.forward(attention_mask=mask, position_ids=positions,
                               inputs_embeds=[prefix, suffix], use_cache=False)
    joint[1].square().sum().backward()
    gradients = {name: p.grad.clone() for name, p in core.named_parameters() if p.requires_grad}
    suffix_gradient = suffix.grad.clone()
    core.zero_grad(set_to_none=True)
    suffix.grad = None
    with torch.no_grad():
        _, cache, _ = core.forward(attention_mask=mask[:, :4, :4], position_ids=positions[:, :, :4],
                                   inputs_embeds=[prefix, None], use_cache=True, fill_kv_cache=True)
    split, _, _ = core.forward(attention_mask=mask[:, 4:], position_ids=positions[:, :, 4:],
                               inputs_embeds=[None, suffix], past_key_values=cache,
                               use_cache=True, fill_kv_cache=False)
    torch.testing.assert_close(split[1], joint[1], rtol=1e-5, atol=1e-6)
    split[1].square().sum().backward()
    torch.testing.assert_close(suffix.grad, suffix_gradient, rtol=1e-4, atol=1e-6)
    for name, gradient in gradients.items():
        torch.testing.assert_close(dict(core.named_parameters())[name].grad, gradient, rtol=1e-4, atol=1e-6)


def test_tiny_bc_resume_and_merged_export(tmp_path):
    report = run_smoke(tmp_path, "cpu", 30)
    assert report["final_loss"] < report["initial_loss"] * 0.2
    assert report["merged_export_reload"] and report["frozen_weights_unchanged"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_tiny_cuda_bf16_bc_resume_and_export(tmp_path):
    report = run_smoke(tmp_path, "cuda", 30)
    assert report["final_loss"] < report["initial_loss"]
