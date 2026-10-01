"""Runtime compatibility adapters; upstream source files remain untouched."""
from __future__ import annotations

from contextlib import contextmanager
from functools import partial
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import torch
from torch.utils.checkpoint import checkpoint

from .training_checkpoint import load_safetensors_streamed
from .training_models import fused_storage_torch_forward, sdpa_attention


def add_vendor_path(root):
    vendor = Path(root) / "vendor" / "lingbot-vla-v2"
    if not (vendor / "lingbotvla").is_dir():
        raise FileNotFoundError(f"Official LingBot-VLA v2 checkout missing: {vendor}")
    if str(vendor) not in sys.path:
        sys.path.insert(0, str(vendor))
    install_reference_moe_import()
    return vendor


def install_reference_moe_import():
    """Avoid upstream group-GEMM's CUDA probe merely to import the model.

    The vendor ops package eagerly imports its distributed kernels, whose
    decorators query a GPU at import time. Local training deliberately uses
    the native Torch MoE backend, including CPU meta/architecture checks.
    Provide that same backend through the vendor function's API before the
    package loads; no kernel files or official checkpoint keys are changed.
    """
    name = "lingbotvla.ops.fused_moe"
    if name in sys.modules:
        return
    module = ModuleType(name)

    def reference_forward(module, num_experts, routing_weights, selected_experts,
                          hidden_states, fc1_1_weight, fc1_2_weight, fc2_weight):
        weights = SimpleNamespace(gate_proj=fc1_1_weight, up_proj=fc1_2_weight, down_proj=fc2_weight)
        return fused_storage_torch_forward(weights, module, num_experts, routing_weights,
                                           selected_experts, hidden_states)

    module.fused_moe_forward = reference_forward
    sys.modules[name] = module


@contextmanager
def sdpa_construction():
    """Override upstream hard-coded FlashAttention settings while constructing.

    Nested VLM/action constructors set FA2 unconditionally. A vision-only SDPA
    config override is insufficient; the Transformers factory must also see SDPA.
    """
    from transformers import PreTrainedModel
    descriptor = PreTrainedModel.__dict__["_from_config"]
    original = PreTrainedModel._from_config.__func__

    def factory(cls, config, **kwargs):
        config._attn_implementation = "sdpa"
        for key in ("text_config", "vision_config"):
            if hasattr(config, key):
                getattr(config, key)._attn_implementation = "sdpa"
        kwargs["attn_implementation"] = "sdpa"
        kwargs["torch_dtype"] = torch.bfloat16
        return original(cls, config, **kwargs)

    PreTrainedModel._from_config = classmethod(factory)
    try:
        yield
    finally:
        PreTrainedModel._from_config = descriptor


def build_policy(root, architecture, base_checkpoint, qwen_path, *, device="cuda", load_weights=True,
                 policy_class=None, weight_dtype=torch.bfloat16):
    add_vendor_path(root)
    from accelerate import init_empty_weights
    from lingbotvla.models.vla.lingbot_vla.configuration_lingbot_vla import LingbotVLAV2Config
    from lingbotvla.models.vla.lingbot_vla.modeling_lingbot_vla_v2 import LingbotVlaV2Policy
    from lingbotvla.models.vla.lingbot_vla.qwen3vl_in_vla import apply_lingbot_qwen3_vl_patch
    from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import apply_lingbot_qwen2_patch, Qwen2FusedExperts
    from lingbotvla.models.vla.lingbot_vla import qwen2_action_expert
    apply_lingbot_qwen3_vl_patch()
    apply_lingbot_qwen2_patch()
    # Evaluation has an inference-only Triton fast path that bypasses experts.forward.
    # Disable it as well so train/validation/local serving share the reference backend.
    qwen2_action_expert.robby_moe_forward = None
    values = {**architecture["model"], **architecture["train"]}
    values.update(tokenizer_path=str(qwen_path), train_expert_only=True, freeze_vision_encoder=True,
                  use_compile=False, vit_attn_implementation="sdpa", attention_implementation="eager",
                  action_fp32=True, post_training=True, moe_implementation="fused")
    config = LingbotVLAV2Config(**values)
    # Keep query-token/head structure exactly as in the pretrained architecture.
    # Teacher weights in align_params are metadata only; our BC loss never loads them.
    with init_empty_weights(include_buffers=False), sdpa_construction():
        policy = (policy_class or LingbotVlaV2Policy)(config, eval=False)
    if not load_weights:
        # MoE/auxiliary modules are created outside nested HF factories and
        # otherwise default to FP32 even on meta, distorting the memory report.
        policy.to(dtype=weight_dtype)
        return policy
    if str(device).startswith("cuda"):
        weight_gib = sum(p.numel() for p in policy.parameters()) * torch.empty((), dtype=weight_dtype).element_size() / 2**30
        gpu_gib = torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30
        if weight_gib > gpu_gib * 0.92:
            raise RuntimeError(f"Model weights alone need {weight_gib:.2f} GiB at {weight_dtype}; GPU has {gpu_gib:.2f} GiB. Select bf16 or use a larger server GPU.")
    load_safetensors_streamed(policy, base_checkpoint, dtype=weight_dtype, device=device)
    # Nonpersistent routing buffers were initialized on CPU by accelerate.
    policy.to(device)
    coupled = policy.model.qwenvl_with_expert
    coupled.attention_interface = partial(sdpa_attention, compute_dtype=weight_dtype)
    # Retain official fused storage but use native PyTorch autograd for local runs.
    for module in policy.modules():
        if isinstance(module, Qwen2FusedExperts):
            module.forward = fused_storage_torch_forward.__get__(module, type(module))
    return policy


def enable_action_checkpointing(model):
    """Checkpoint the actual two-stage expert calls used by the coupled forward.

    Setting the top-level HF gradient_checkpointing flag alone does not reliably
    wrap this project's custom compute_kqv/output_atten decoder interface.
    """
    expert = model.model.qwenvl_with_expert.qwen_expert.model
    for layer in expert.layers:
        if getattr(layer, "_lrvla_checkpointed", False):
            continue
        original = layer.forward

        def wrapped(*args, _layer=layer, _original=original, **kwargs):
            if _layer.training and torch.is_grad_enabled():
                return checkpoint(_original, *args, use_reentrant=False, **kwargs)
            return _original(*args, **kwargs)

        layer.forward = wrapped
        layer._lrvla_checkpointed = True
