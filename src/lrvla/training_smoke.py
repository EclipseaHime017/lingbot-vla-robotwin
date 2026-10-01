"""Tiny synthetic regression check: optimization, resume and merged export.

This validates the local machinery only, never a 6B model or RoboTwin success rate.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn

from .training_checkpoint import export_hf_checkpoint, load_safetensors_streamed, resume_training_checkpoint, save_training_checkpoint
from .training_models import configure_trainable, masked_bc_loss, memory_report


class TinyConfig(SimpleNamespace):
    def to_json_file(self, path):
        Path(path).write_text(json.dumps(vars(self), indent=2))


class TinyAttention(nn.Module):
    def __init__(self, dim=8):
        super().__init__()
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            setattr(self, name, nn.Linear(dim, dim, bias=False))

    def forward(self, x):
        return self.o_proj(torch.tanh((self.q_proj(x) + self.k_proj(x) + self.v_proj(x)) / 3))


class TinyLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attn = TinyAttention()

    def forward(self, x):
        return x + self.self_attn(x)


class TinyExpert(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TinyLayer(), TinyLayer()])
        self.norm = nn.LayerNorm(8)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return self.norm(x)


class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = TinyConfig(model_type="synthetic_smoke", action_dim=2)
        self.model = nn.Module()
        self.model.qwenvl_with_expert = nn.Module()
        self.model.qwenvl_with_expert.qwenvl = nn.Linear(8, 8)
        self.model.qwenvl_with_expert.qwen_expert = nn.Module()
        self.model.qwenvl_with_expert.qwen_expert.model = TinyExpert()
        self.model.state_proj = nn.Linear(2, 8)
        self.model.action_in_proj = nn.Linear(2, 8)
        self.model.action_out_proj = nn.Linear(8, 2)
        self.model.action_time_mlp_in = nn.Linear(16, 8)
        self.model.action_time_mlp_out = nn.Linear(8, 8)

    def forward(self, state):
        x = self.model.state_proj(state)
        x = self.model.qwenvl_with_expert.qwen_expert.model(x)
        return self.model.action_out_proj(x)[:, None]


def run_smoke(output, device="cpu", steps=30):
    if steps < 2:
        raise ValueError("Synthetic smoke needs at least 2 optimizer steps.")
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")
    torch.manual_seed(7)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    model = TinyPolicy().to(device)
    selection = configure_trainable(model, {"mode": "lora", "lora_rank": 2, "lora_alpha": 4})
    frozen = {name: p.detach().clone() for name, p in model.named_parameters() if not p.requires_grad}
    x = torch.randn(32, 2, device=device)
    y = torch.stack([x[:, 0] * 0.7 - x[:, 1] * 0.3, x[:, 0] * 0.2 + x[:, 1] * 0.5], dim=-1)[:, None]
    mask = torch.ones_like(y, dtype=torch.bool)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.03, foreach=False)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    initial = float(masked_bc_loss(model(x), y, mask, "fm").detach())
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device, dtype=torch.bfloat16, enabled=device == "cuda"):
            loss = masked_bc_loss(model(x), y, mask, "fm")
        loss.backward()
        optimizer.step()
        scheduler.step()
    model.eval()
    prediction = model(x).detach()
    final = float(masked_bc_loss(prediction, y, mask, "fm"))
    if final >= initial:
        raise AssertionError(f"Tiny BC regression did not improve: {initial} -> {final}")
    for name, before in frozen.items():
        if not torch.equal(before, dict(model.named_parameters())[name]):
            raise AssertionError(f"Frozen parameter changed: {name}")
    metadata = {"base_checkpoint": "synthetic", "base_identity": {"synthetic": True},
                "manifest_sha256": "synthetic", "manifest_content_sha256": "synthetic", "selection": selection}
    save_training_checkpoint(model, optimizer, scheduler, output / "training.pt", {"global_step": steps}, metadata)
    reloaded = TinyPolicy().to(device)
    configure_trainable(reloaded, {"mode": "lora", "lora_rank": 2, "lora_alpha": 4})
    # Frozen base weights represent the same downloaded base at resume time.
    with torch.no_grad():
        for name, value in frozen.items():
            dict(reloaded.named_parameters())[name].copy_(value)
    resumed_optimizer = torch.optim.AdamW([p for p in reloaded.parameters() if p.requires_grad], lr=0.03, foreach=False)
    resumed_scheduler = torch.optim.lr_scheduler.LambdaLR(resumed_optimizer, lambda _: 1.0)
    state = resume_training_checkpoint(reloaded, resumed_optimizer, resumed_scheduler, output / "training.pt", expected_metadata=metadata)
    reloaded.eval()
    torch.testing.assert_close(reloaded(x), prediction, rtol=0, atol=0)
    exported = export_hf_checkpoint(model, output / "hf_ckpt", dtype=torch.float32, max_shard_bytes=1024)
    native = TinyPolicy().to(device)
    load_safetensors_streamed(native, exported, dtype=torch.float32)
    native.to(device).eval()
    torch.testing.assert_close(native(x), prediction, rtol=1e-5, atol=1e-6)
    report = {"synthetic_only": True, "device": device, "steps": steps,
              "initial_loss": initial, "final_loss": final, "resume_step": state["global_step"],
              "frozen_weights_unchanged": True, "merged_export_reload": True, "memory": memory_report(model)}
    (output / "result.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report), flush=True)
    return report
