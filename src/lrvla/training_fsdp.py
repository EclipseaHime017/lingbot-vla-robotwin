"""FSDP adapter for a frozen VLM and a fully trainable action expert."""
from __future__ import annotations

from functools import partial

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel, ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

from .training_models import flow_matching_bc


class BCForward(nn.Module):
    """Put the complete BC computation inside FSDP's forward hooks.

    The root hooks gather projection parameters before the action head reads
    weights directly. Optional child wrappers gather each expert block for its
    two upstream calls, computing Q/K/V and the post-attention output.
    """

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, batch, *, reduction="mean"):
        return flow_matching_bc(self.policy, batch, reduction=reduction)


def fsdp_parameter_groups(policy):
    """Return FP32 trainables and the frozen parameters FSDP must ignore."""
    trainable, frozen = [], []
    for name, parameter in policy.named_parameters():
        if parameter.requires_grad:
            if parameter.dtype != torch.float32:
                raise ValueError(f"FSDP expert training requires FP32 trainable parameters: {name} is {parameter.dtype}.")
            trainable.append(parameter)
        else:
            frozen.append(parameter)
    if not trainable:
        raise ValueError("FSDP training requires at least one trainable parameter.")
    return trainable, frozen


def fsdp_shard_report(policy, *, layer_sharding=True):
    """Count the largest trainable group before FSDP changes parameter shapes."""
    trainable = {id(parameter): parameter for parameter in policy.parameters() if parameter.requires_grad}
    total = sum(parameter.numel() for parameter in trainable.values())
    layer_groups, layer_parameters = [], set()
    if layer_sharding:
        layers = policy.model.qwenvl_with_expert.qwen_expert.model.layers
        for layer in layers:
            parameters = {id(parameter): parameter for parameter in layer.parameters() if parameter.requires_grad}
            layer_groups.append(sum(parameter.numel() for parameter in parameters.values()))
            layer_parameters.update(parameters)
    root = sum(parameter.numel() for identity, parameter in trainable.items() if identity not in layer_parameters)
    return {
        "total_trainable_parameters": total, "root_trainable_parameters": root,
        "decoder_layer_trainable_parameters": layer_groups,
        "largest_shard_parameters": max([root, *layer_groups]),
        "wrapped_decoder_layers": len(layer_groups),
    }


def fsdp_largest_shard_parameters(policy, *, layer_sharding=True):
    return fsdp_shard_report(policy, layer_sharding=layer_sharding)["largest_shard_parameters"]


def wrap_expert_fsdp(policy, *, device, process_group=None, layer_sharding=False):
    """Shard trainable parameters, optionally wrapping individual decoder layers.

    Call after configure_trainable(..., mode='expert') and moving the policy to
    this rank's CUDA device. Frozen BF16 VLM parameters stay replicated and are
    excluded from flattening; every managed parameter remains FP32. CUDA
    autocast controls compute precision in the caller.
    """
    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Expert FSDP training requires a CUDA device.")
    if not torch.distributed.is_initialized():
        raise RuntimeError("Initialize the distributed process group before constructing FSDP.")
    if device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    _, frozen = fsdp_parameter_groups(policy)
    for name, parameter in policy.named_parameters():
        if parameter.device != device:
            raise ValueError(f"Move the entire policy to {device} before FSDP wrapping: {name} is on {parameter.device}.")
    options = {}
    if layer_sharding:
        from lingbotvla.models.vla.lingbot_vla.qwen2_action_expert import Qwen2DecoderLayer
        # The actual upstream coupled loop calls each block twice (Q/K/V and
        # post-attention). The two-rank FULL_SHARD regression verifies both
        # calls with checkpoint recomputation, averaged gradients and Adam.
        options["auto_wrap_policy"] = partial(transformer_auto_wrap_policy,
                                               transformer_layer_cls={Qwen2DecoderLayer})
    return FullyShardedDataParallel(
        BCForward(policy), process_group=process_group,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        ignored_states=frozen, use_orig_params=True, device_id=device,
        limit_all_gathers=True, **options,
    )
