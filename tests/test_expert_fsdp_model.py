"""Verify frozen BF16 parameters are excluded from FP32 expert sharding."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import pytest
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from lrvla import training_fsdp
from lrvla.training_fsdp import BCForward, fsdp_parameter_groups, wrap_expert_fsdp
from lrvla.training_models import configure_trainable
from lrvla.training_smoke import TinyPolicy


def test_entire_expert_is_managed_and_frozen_bf16_vlm_is_ignored():
    policy = TinyPolicy().to(torch.bfloat16)
    configure_trainable(policy, {"mode": "expert", "expert_last_n_layers": -1})
    trainable, frozen = fsdp_parameter_groups(policy)
    assert {id(p) for p in trainable}.isdisjoint(id(p) for p in frozen)
    assert len(trainable) + len(frozen) == len(list(policy.parameters()))
    assert all(p.dtype == torch.float32 and p.requires_grad for p in trainable)
    assert all(p.dtype == torch.bfloat16 and not p.requires_grad for p in frozen)
    assert {id(p) for p in policy.model.qwenvl_with_expert.qwenvl.parameters()} <= {id(p) for p in frozen}
    assert {id(p) for p in policy.model.qwenvl_with_expert.qwen_expert.model.parameters()} <= {id(p) for p in trainable}


def test_no_trainable_or_low_precision_trainable_selection_fails_early():
    policy = nn.Linear(2, 2).requires_grad_(False)
    with pytest.raises(ValueError, match="at least one"):
        fsdp_parameter_groups(policy)
    policy.requires_grad_(True).to(torch.bfloat16)
    with pytest.raises(ValueError, match="FP32 trainable"):
        fsdp_parameter_groups(policy)


def test_bc_forward_calls_existing_loss_inside_module_forward(monkeypatch):
    policy = nn.Linear(2, 2)
    observations = {"state": torch.randn(2, 2)}
    calls = []
    def loss(model, batch, *, reduction):
        calls.append((model, batch, reduction))
        return model(batch["state"]).square().sum()
    monkeypatch.setattr(training_fsdp, "flow_matching_bc", loss)
    wrapped = BCForward(policy)
    hooks = []
    wrapped.register_forward_pre_hook(lambda *args: hooks.append("pre"))
    wrapped.register_forward_hook(lambda *args: hooks.append("post"))
    wrapped(observations, reduction="sum").backward()
    assert calls == [(policy, observations, "sum")]
    assert hooks == ["pre", "post"]
    assert policy.weight.grad is not None


def test_wrapping_without_cuda_fails_before_process_group_initialization():
    with pytest.raises(ValueError, match="CUDA"):
        wrap_expert_fsdp(nn.Linear(2, 2), device="cpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FSDP requires a CUDA device")
def test_cuda_fsdp_keeps_frozen_prefix_and_recomputed_expert_gradients(tmp_path, monkeypatch):
    """Real FSDP hooks, FP32 managed state, ignored BF16 VLM, and checkpoints.

    One local GPU validates integration. Four/eight-rank sharding and NCCL are
    exercised by the actual server run, rather than simulated by this test.
    """
    if torch.distributed.is_initialized():
        pytest.skip("A process group is already active in this process.")
    torch.distributed.init_process_group("gloo", init_method=(tmp_path / "rendezvous").as_uri(), rank=0, world_size=1)
    try:
        device = torch.device("cuda", 0)
        torch.manual_seed(123)
        policy = TinyPolicy().to(device=device, dtype=torch.bfloat16)
        configure_trainable(policy, {"mode": "expert", "expert_last_n_layers": -1})
        # Mimic the action output projection's upstream direct parameter access.
        def loss(model, batch, *, reduction):
            x = model.model.state_proj(batch["state"])
            for layer in model.model.qwenvl_with_expert.qwen_expert.model.layers:
                x = checkpoint(layer, x, use_reentrant=False)
            norm = model.model.qwenvl_with_expert.qwen_expert.model.norm
            x = norm(x)
            head = model.model.action_out_proj
            prediction = torch.nn.functional.linear(x.float(), head.weight.float(), head.bias.float())
            return (prediction.float() - batch["target"]).square().sum()
        monkeypatch.setattr(training_fsdp, "flow_matching_bc", loss)
        batch = {"state": torch.randn(4, 2, device=device), "target": torch.randn(4, 2, device=device)}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss(policy, batch, reduction="sum").backward()
        expected = {name: p.grad.clone() if p.grad is not None else None for name, p in policy.named_parameters()}
        policy.zero_grad(set_to_none=True)
        frozen_before = {name: p.detach().clone() for name, p in policy.named_parameters() if not p.requires_grad}
        distributed = wrap_expert_fsdp(policy, device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            distributed(batch, reduction="sum").backward()
        for name, parameter in policy.named_parameters():
            if expected[name] is None:
                if not parameter.requires_grad:
                    assert parameter.grad is None
                else:
                    # Flat gradients allocate zero views for unused managed
                    # parameters, while ignored frozen parameters remain None.
                    assert parameter.grad is None or not parameter.grad.count_nonzero()
            else:
                assert parameter.dtype == torch.float32
                torch.testing.assert_close(parameter.grad, expected[name], rtol=1e-4, atol=1e-5)
        optimizer = torch.optim.AdamW([p for p in distributed.parameters() if p.requires_grad], lr=1e-3)
        optimizer.step()
        for name, before in frozen_before.items():
            torch.testing.assert_close(dict(policy.named_parameters())[name], before, rtol=0, atol=0)
    finally:
        torch.distributed.destroy_process_group()
