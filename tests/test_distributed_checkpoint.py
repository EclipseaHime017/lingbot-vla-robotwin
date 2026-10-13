"""Distributed checkpoint identity, exact resume and complete HF export."""
from pathlib import Path
import hashlib
import json
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np
import pytest
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from lrvla import training_fsdp
from lrvla.training_distributed_checkpoint import (
    export_distributed_hf, resume_distributed_checkpoint, save_distributed_checkpoint,
)
from lrvla.training_fsdp import BCForward, wrap_expert_fsdp
from lrvla.training_models import configure_trainable
from lrvla.training_smoke import TinyPolicy
from lrvla.training_checkpoint import load_safetensors_streamed


@pytest.fixture
def process_group(tmp_path):
    if dist.is_initialized():
        pytest.skip("A process group is already active in this process.")
    try:
        dist.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    except RuntimeError as error:
        if "Cannot resolve 127.0.0.1" in str(error):
            pytest.skip("The execution sandbox blocks Gloo's local network interface.")
        raise
    try:
        yield
    finally:
        dist.destroy_process_group()


def prepare_model(device, monkeypatch):
    torch.manual_seed(10)
    policy = TinyPolicy().to(device=device, dtype=torch.bfloat16)
    selection = configure_trainable(policy, {"mode": "expert", "expert_last_n_layers": -1})
    def supervised_bc(model, batch, *, reduction):
        # Include all five projection modules, as the actual BC suffix does.
        flow = model.model
        state, action = flow.state_proj(batch["state"]), flow.action_in_proj(batch["state"])
        x = state + action + flow.action_time_mlp_out(torch.tanh(
            flow.action_time_mlp_in(torch.cat([state, action], dim=-1))))
        x = flow.qwenvl_with_expert.qwen_expert.model(x)
        prediction = flow.action_out_proj(x)[:, None]
        loss = (prediction.float() - batch["target"]).square()
        return loss.sum() if reduction == "sum" else loss.mean()
    monkeypatch.setattr(training_fsdp, "flow_matching_bc", supervised_bc)
    model = wrap_expert_fsdp(policy, device=device) if device.type == "cuda" else BCForward(policy)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-3, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: 1 / (step + 1))
    metadata = {"base_identity": {"sha256": "fixed-base"}, "manifest_content_sha256": "clean-manifest",
                "selection": selection, "norm_sha256": "fixed-norm", "model_geometry": {"action_dim": 2},
                "qwen_config_sha256": "fixed-qwen", "loss_normalization": "valid-global-v2",
                "data_sampling": {"world_size": 1, "seed": 42}}
    return policy, model, optimizer, scheduler, metadata


def update(model, optimizer, scheduler, observations):
    optimizer.zero_grad(set_to_none=True)
    model(observations, reduction="sum").backward()
    optimizer.step()
    scheduler.step()


def sample_rng(device):
    result = [torch.rand(3), random.random(), np.random.rand(3)]
    if device.type == "cuda":
        result.append(torch.rand(3, device=device))
    return result


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_collective_checkpoint_resumes_optimizer_scheduler_and_rank_rng(
        tmp_path, process_group, monkeypatch, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("FSDP requires a CUDA device.")
    device = torch.device(device_name, 0) if device_name == "cuda" else torch.device("cpu")
    policy, model, optimizer, scheduler, metadata = prepare_model(device, monkeypatch)
    observations = {"state": torch.randn(4, 2, device=device), "target": torch.randn(4, 1, 2, device=device)}
    update(model, optimizer, scheduler, observations)
    checkpoint = tmp_path / "checkpoint"
    save_distributed_checkpoint(model, optimizer, scheduler, checkpoint, {"global_step": 1, "epoch": 2}, metadata,
                                rank=0, world_size=1, device=device)
    expected_rng = sample_rng(device)
    update(model, optimizer, scheduler, observations)
    expected = {name: p.detach().clone() for name, p in policy.named_parameters()}
    expected_lr = scheduler.get_last_lr()

    restored_policy, restored, restored_optimizer, restored_scheduler, restored_metadata = prepare_model(device, monkeypatch)
    state = resume_distributed_checkpoint(restored, restored_optimizer, restored_scheduler, checkpoint,
                                         expected_metadata=restored_metadata, rank=0, world_size=1, device=device)
    assert state == {"global_step": 1, "epoch": 2}
    actual_rng = sample_rng(device)
    for expected_value, actual_value in zip(expected_rng, actual_rng):
        if isinstance(expected_value, torch.Tensor):
            torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
        elif isinstance(expected_value, np.ndarray):
            np.testing.assert_array_equal(actual_value, expected_value)
        else:
            assert actual_value == expected_value
    update(restored, restored_optimizer, restored_scheduler, observations)
    assert restored_scheduler.get_last_lr() == expected_lr
    for name, parameter in restored_policy.named_parameters():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    assert (checkpoint / "complete.json").is_file()
    assert (checkpoint / "rank-00000.pt").is_file()
    stored_keys = dcp.FileSystemReader(checkpoint / "shards").read_metadata().state_dict_metadata
    assert not any("qwenvl." in name for name in stored_keys)


def test_incomplete_or_changed_identity_fails_before_mutating_parameters(tmp_path, process_group, monkeypatch):
    device = torch.device("cpu")
    policy, model, optimizer, scheduler, metadata = prepare_model(device, monkeypatch)
    checkpoint = tmp_path / "checkpoint"
    before = {name: p.detach().clone() for name, p in policy.named_parameters()}
    with pytest.raises(ValueError, match="incomplete"):
        resume_distributed_checkpoint(model, optimizer, scheduler, checkpoint,
                                      expected_metadata=metadata, rank=0, world_size=1, device=device)
    save_distributed_checkpoint(model, optimizer, scheduler, checkpoint, {"global_step": 0}, metadata,
                                rank=0, world_size=1, device=device)
    for key in ("base_identity", "manifest_content_sha256", "data_sampling"):
        changed = {**metadata, key: "changed"}
        with pytest.raises(ValueError, match=key):
            resume_distributed_checkpoint(model, optimizer, scheduler, checkpoint,
                                          expected_metadata=changed, rank=0, world_size=1, device=device)
    with pytest.raises(FileExistsError, match="overwrite"):
        save_distributed_checkpoint(model, optimizer, scheduler, checkpoint, {"global_step": 0}, metadata,
                                    rank=0, world_size=1, device=device)
    run_path, complete_path = checkpoint / "run_state.json", checkpoint / "complete.json"
    run_bytes, complete_bytes = run_path.read_bytes(), complete_path.read_bytes()
    changed_run, changed_complete = json.loads(run_bytes), json.loads(complete_bytes)
    changed_run["world_size"] = changed_complete["world_size"] = 2
    run_path.write_text(json.dumps(changed_run))
    changed_complete["run_state_sha256"] = hashlib.sha256(run_path.read_bytes()).hexdigest()
    complete_path.write_text(json.dumps(changed_complete))
    with pytest.raises(ValueError, match="world_size"):
        resume_distributed_checkpoint(model, optimizer, scheduler, checkpoint,
                                      expected_metadata=metadata, rank=0, world_size=1, device=device)
    run_path.write_bytes(run_bytes)
    complete_path.write_bytes(complete_bytes)
    for name, parameter in policy.named_parameters():
        torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)


@pytest.mark.parametrize("device_name", ["cpu", "cuda"])
def test_export_contains_frozen_vlm_and_updated_expert_with_official_keys(
        tmp_path, process_group, monkeypatch, device_name):
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("FSDP requires a CUDA device.")
    device = torch.device(device_name, 0) if device_name == "cuda" else torch.device("cpu")
    policy, model, optimizer, scheduler, metadata = prepare_model(device, monkeypatch)
    observations = {"state": torch.randn(4, 2, device=device), "target": torch.randn(4, 1, 2, device=device)}
    update(model, optimizer, scheduler, observations)
    expected = {name: value.detach().cpu().to(torch.bfloat16) for name, value in policy.state_dict().items()}
    export_path = export_distributed_hf(model, policy, tmp_path / "hf_ckpt", dtype=torch.bfloat16,
                                        processor=None, rank=0)
    reloaded = TinyPolicy()
    load_safetensors_streamed(reloaded, export_path, dtype=torch.bfloat16, device="cpu")
    for name, parameter in reloaded.state_dict().items():
        torch.testing.assert_close(parameter, expected[name], rtol=0, atol=0)
    assert any("qwenvl." in name for name in expected)
