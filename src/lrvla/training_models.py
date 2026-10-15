"""Small, independently testable adaptations for clean-only behaviour cloning."""
from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


class LoRALinear(nn.Module):
    """Keep official base weights frozen and learn a zero-initialized residual."""

    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16, dropout: float = 0):
        super().__init__()
        if rank <= 0 or not 0 <= dropout < 1:
            raise ValueError("LoRA rank must be positive and dropout must be in [0, 1).")
        self.base = base.requires_grad_(False)
        self.rank, self.alpha = rank, alpha
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout)
        self.lora_A = nn.Parameter(torch.empty(rank, base.in_features, device=base.weight.device, dtype=torch.float32))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device, dtype=torch.float32))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    @property
    def weight(self):
        # Upstream occasionally inspects a projection's dtype directly.
        return self.base.weight

    @property
    def bias(self):
        return self.base.bias

    def forward(self, x):
        base_out = self.base(x)
        with torch.autocast(device_type=x.device.type, enabled=False):
            residual = F.linear(F.linear(self.dropout(x.float()), self.lora_A), self.lora_B)
        return base_out + residual.to(base_out.dtype) * self.scale

    def merged_weight(self):
        # Export computes this on CPU to avoid another copy of the 6B GPU model.
        return self.base.weight.detach().float().cpu() + (
            self.lora_B.detach().float().cpu() @ self.lora_A.detach().float().cpu()
        ) * self.scale


ACTION_ROOT = "model.qwenvl_with_expert.qwen_expert.model"
PROJECTIONS = (
    "model.state_proj", "model.action_in_proj", "model.action_out_proj",
    "model.action_time_mlp_in", "model.action_time_mlp_out",
)


def configure_trainable(model, config):
    """Freeze VLM and select action LoRA, partial expert, or full expert."""
    model.requires_grad_(False)
    mode = config["mode"]
    if mode not in {"lora", "expert"}:
        raise ValueError("mode must be lora or expert; competition training is BC only.")
    expert = model.get_submodule(ACTION_ROOT)
    names = []
    if mode == "lora":
        targets = set(config.get("lora_targets", ["q_proj", "k_proj", "v_proj", "o_proj"]))
        for name, module in list(expert.named_modules()):
            if name.rsplit(".", 1)[-1] in targets and isinstance(module, nn.Linear):
                parent_name, _, child = name.rpartition(".")
                parent = expert.get_submodule(parent_name) if parent_name else expert
                setattr(parent, child, LoRALinear(module, config.get("lora_rank", 8),
                                                config.get("lora_alpha", 16), config.get("lora_dropout", 0)))
                names.append(f"{ACTION_ROOT}.{name}")
        if not names:
            raise ValueError("No action expert Linear modules matched lora_targets.")
    else:
        n = int(config.get("expert_last_n_layers", 2))
        if n != -1 and not 1 <= n <= len(expert.layers):
            raise ValueError("expert_last_n_layers must be -1 (all) or between 1 and layer count.")
        layers = expert.layers if n == -1 else expert.layers[-n:]
        for layer in layers:
            layer.requires_grad_(True)
        expert.norm.requires_grad_(True)
    if config.get("train_projections", True):
        for path in PROJECTIONS:
            model.get_submodule(path).requires_grad_(True)
    # AdamW moments use parameter dtype: explicitly retain FP32 trainables.
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable_names = [name for name, p in model.named_parameters() if p.requires_grad]
    if not trainable_names:
        raise ValueError("Training selection contains no parameters.")
    if any("qwenvl." in name for name in trainable_names):
        raise AssertionError("The VLM must remain frozen for the local action-only presets.")
    return {"mode": mode, "lora_modules": names, "trainable_names": trainable_names,
            "expert_last_n_layers": config.get("expert_last_n_layers", 2) if mode == "expert" else None,
            "lora": {"rank": config.get("lora_rank", 8), "alpha": config.get("lora_alpha", 16),
                     "dropout": config.get("lora_dropout", 0)} if mode == "lora" else None}


def memory_report(model):
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    weights = sum(p.numel() * p.element_size() for p in model.parameters())
    # FP32 gradients + two FP32 AdamW moment tensors; excludes activations/caches/workspace.
    lower_bound = weights + trainable * 12
    return {"total_parameters": total, "trainable_parameters": trainable,
            "weight_gib": weights / 2**30, "adamw_training_lower_bound_gib": lower_bound / 2**30,
            "excludes": "activations, VLM KV caches, temporary tensors, CUDA context, allocator fragmentation"}


def sdpa_attention(query, key, value, attention_mask, *, compute_dtype=None):
    """Official [B,L,H,D] attention interface using PyTorch SDPA with GQA."""
    if query.shape[2] % key.shape[2]:
        raise ValueError("query heads must be a multiple of key/value heads")
    q, k, v = (t.transpose(1, 2) for t in (query, key, value))
    # Qwen rotary computations deliberately use FP32; bf16 SDPA reduces storage.
    dtype = compute_dtype or q.dtype
    q, k, v = (t.to(dtype) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, attention_mask[:, None].bool(),
                                         dropout_p=0.0, is_causal=False, enable_gqa=q.shape[1] != k.shape[1])
    return out.transpose(1, 2).contiguous().flatten(2)


def fused_storage_torch_forward(self, module, num_experts, routing_weights, selected_experts, hidden_states):
    """Autograd-compatible reference MoE retaining official fused tensor keys.

    Avoids distributed initialization and compilation of group GEMM extensions.
    Slower than official fused kernels; suitable for validating a local baseline.
    """
    del module
    out = torch.zeros_like(hidden_states)
    for idx in range(num_experts):
        token, slot = torch.where(selected_experts == idx)
        if token.numel() == 0:
            continue
        x = hidden_states[token]
        up = F.linear(x, self.up_proj[idx])
        gate = F.silu(F.linear(x, self.gate_proj[idx]))
        y = F.linear(up * gate, self.down_proj[idx])
        out = out.index_add(0, token, y * routing_weights[token, slot, None])
    return out


def masked_bc_loss(prediction, target, joint_mask, loss_type="L1_fm", *, reduction="mean"):
    if prediction.shape != target.shape or joint_mask.shape != target.shape:
        raise ValueError("Prediction, target and joint_mask must have identical [B,T,D] shapes.")
    mask = joint_mask.to(torch.float32)
    count = mask.sum()
    if count.item() <= 0:
        raise ValueError("Batch has no valid action dimensions.")
    if loss_type == "L1_fm":
        loss = (prediction.float() - target.float()).abs()
    elif loss_type == "fm":
        loss = (prediction.float() - target.float()).square()
    else:
        raise ValueError(f"Unsupported BC flow-matching loss: {loss_type}")
    numerator = (loss * mask).sum()
    if reduction == "sum":
        return numerator
    if reduction == "mean":
        return numerator / count
    raise ValueError(f"Unsupported BC loss reduction: {reduction}")


@torch.no_grad()
def normalize_bc_gradients(parameters, valid_count):
    """Normalize summed microbatch gradients before clipping/AdamW.

    The denominator covers the whole update window, including short terminal
    chunks and a partial batch at an epoch boundary.
    """
    if not math.isfinite(valid_count) or valid_count <= 0:
        raise ValueError("Accumulation window must contain a finite positive valid action count.")
    for parameter in parameters:
        if parameter.grad is not None:
            parameter.grad.div_(valid_count)


def configure_moe_loss(model, config):
    values = {key: float(config.get(key, getattr(model.config, key, default))) for key, default in
              (("sequence_wise_loss_coeff", 1e-3), ("router_z_loss_coeff", 1e-4))}
    if any(not math.isfinite(value) or value < 0 for value in values.values()):
        raise ValueError("MoE loss coefficients must be finite and nonnegative.")
    if getattr(model.config, "sequence_wise_mode", "per_sequence") != "per_sequence":
        raise ValueError("Resumable BC supports per_sequence MoE balance only.")
    for key, value in values.items():
        setattr(model.config, key, value)
        setattr(model.model.config, key, value)
    return {**values, "normalization": "sample_mean_v1", "sequence_wise_mode": "per_sequence",
            "router_activation": getattr(model.config, "router_activation", "sigmoid"),
            "token_top_k": getattr(model.config, "token_top_k", 4)}


def moe_auxiliary_losses(router_logits, batch_size, *, top_k=4, score_func="sigmoid"):
    """Per-sample official sequence balance/z-loss; counts have no gradient."""
    balances, zs, loads = [], [], {}
    for layer, logits in enumerate(router_logits):
        if logits.ndim != 2 or logits.shape[0] % batch_size:
            raise ValueError("Router logits must contain equally sized per-sample suffixes.")
        values = logits.float().reshape(batch_size, -1, logits.shape[-1])
        _, tokens, experts = values.shape
        if not 1 <= top_k <= experts:
            raise ValueError("Invalid MoE top_k.")
        if score_func == "sigmoid":
            scores = values.sigmoid()
            probs = scores / scores.sum(-1, keepdim=True).clamp_min(torch.finfo(scores.dtype).tiny)
        elif score_func == "softmax":
            probs = values.softmax(-1)
        else:
            raise ValueError(f"Unsupported router activation: {score_func}")
        with torch.no_grad():
            selected = values.topk(top_k, dim=-1).indices.reshape(batch_size, -1)
            counts = values.new_zeros((batch_size, experts)).scatter_add_(
                1, selected, values.new_ones(selected.shape))
            frequency = counts * (experts / (top_k * tokens))
            total = counts.sum(0)
            distribution = total / total.sum()
            loads[f"layer_{layer}"] = {"tokens_per_expert": total.detach(),
                                        "max_mean_load": (total.max() / total.mean()).detach(),
                                        "entropy": (-(distribution * distribution.clamp_min(1e-30).log()).sum()).detach()}
        balances.append((frequency * probs.mean(1)).sum(-1))
        zs.append(values.logsumexp(-1).square().mean(-1))
    if not balances:
        return None, None, {}
    return torch.stack(balances).mean(0), torch.stack(zs).mean(0), loads


def router_load_metrics(counts, *, distributed=False):
    if not counts:
        return {}
    names = sorted(counts)
    totals = torch.stack([counts[name] for name in names])
    if distributed:
        import torch.distributed as dist
        dist.all_reduce(totals)
    result = {}
    for name, total in zip(names, totals):
        probability = total / total.sum()
        result[name] = {"tokens_per_expert": total.cpu().tolist(),
                        "max_mean_load": float(total.max() / total.mean()),
                        "entropy": float(-(probability * probability.clamp_min(1e-30).log()).sum())}
    return result


def flow_matching_bc(model, batch, *, reduction="mean", auxiliary_scale=None,
                     noise=None, time=None, return_details=False):
    """Use official embedding/transformer modules, supervising clean action chunks.

    Query embeddings and alignment heads remain in the architecture and export;
    no depth/video teacher targets or teacher losses are generated locally.
    """
    flow = model.model
    actions, state = batch["actions"].float(), batch["state"].float()
    noise = torch.randn_like(actions) if noise is None else noise.to(actions.device, dtype=torch.float32)
    time = flow.sample_time(actions.shape[0], actions.device).float() if time is None else time.to(actions.device).float()
    if noise.shape != actions.shape or time.shape != (len(actions),):
        raise ValueError("Explicit flow noise/time shapes differ from the action batch.")
    x_t = time[:, None, None] * noise + (1 - time[:, None, None]) * actions
    with torch.no_grad():
        prefix, prefix_pad, prefix_ar, prefix_pos, visual_mask, deepstack = flow.embed_prefix(
            batch["images"], batch["img_masks"], batch["lang_tokens"], batch["lang_masks"],
            image_grid_thw=batch["image_grid_thw"])
    time_emb, suffix, suffix_pad, suffix_ar = flow.embed_suffix(state, x_t, time)
    from lingbotvla.models.vla.lingbot_vla.utils import make_att_2d_masks
    pad = torch.cat([prefix_pad, suffix_pad], dim=1)
    mask = make_att_2d_masks(pad, torch.cat([prefix_ar, suffix_ar], dim=1))
    prefix_len = prefix.shape[1]
    if getattr(flow, "block_future_depth_to_action", False):
        from lingbotvla.models.vla.lingbot_vla.utils import block_suffix_to_fv_
        mask = block_suffix_to_fv_(mask, suffix_row_start=prefix_len, prefix_len=prefix_len,
                                  num_task_tokens=flow.num_task_tokens)
    mask = flow._block_suffix_to_future_video_if_enabled_(mask, suffix_row_start=prefix_len, prefix_len=prefix_len)
    pos = flow._build_full_position_ids(prefix_pos, prefix_pad, suffix_pad)
    coupled = flow.qwenvl_with_expert
    # Prefix tokens cannot depend on action tokens. Cache their frozen K/Vs and
    # omit their intermediate graphs from the action backward pass.
    if mask[:, :prefix_len, prefix_len:].any():
        raise AssertionError("Frozen prefix split requires prefix-to-action attention to be blocked.")
    with torch.no_grad():
        _, cache, _ = coupled.forward(
            attention_mask=mask[:, :prefix_len, :prefix_len], position_ids=pos[:, :, :prefix_len],
            inputs_embeds=[prefix, None], use_cache=True, fill_kv_cache=True,
            visual_pos_masks=visual_mask, deepstack_visual_embeds=deepstack)
    outputs, _, router_logits = coupled.forward(
        attention_mask=mask[:, prefix_len:, :], position_ids=pos[:, :, prefix_len:],
        inputs_embeds=[None, suffix], past_key_values=cache, use_cache=True, fill_kv_cache=False,
        ada_cond=time_emb if getattr(flow.config, "adanorm_time", False) else None)
    suffix_out = outputs[1][:, -flow.config.n_action_steps:]
    prediction = flow._fp32_linear(flow.action_out_proj, suffix_out)
    if prediction.ndim != 3 or prediction.shape != actions.shape or batch["joint_mask"].shape != actions.shape:
        raise ValueError("Prediction, target and joint_mask must have identical [B,T,D] shapes.")
    mask = batch["joint_mask"].float()
    error = prediction.float() - (noise - actions)
    if flow.config.loss_type == "L1_fm":
        error = error.abs()
    elif flow.config.loss_type == "fm":
        error = error.square()
    else:
        raise ValueError(f"Unsupported BC flow-matching loss: {flow.config.loss_type}")
    bc_samples = (error * mask).flatten(1).sum(1)
    counts = mask.flatten(1).sum(1)
    if counts.sum() <= 0:
        raise ValueError("Batch has no valid action dimensions.")
    balance, z, loads = moe_auxiliary_losses(
        router_logits, len(actions), top_k=getattr(flow.config, "token_top_k", 4),
        score_func=getattr(flow.config, "router_activation", "sigmoid"))
    zero = bc_samples.new_zeros(len(actions))
    balance = zero if balance is None else balance * getattr(flow.config, "sequence_wise_loss_coeff", 0)
    z = zero if z is None else z * getattr(flow.config, "router_z_loss_coeff", 0)
    scale = counts.sum() / len(actions) if auxiliary_scale is None else auxiliary_scale
    numerator = bc_samples.sum() + scale * (balance.sum() + z.sum())
    if reduction not in {"mean", "sum"}:
        raise ValueError(f"Unsupported BC loss reduction: {reduction}")
    loss = numerator if reduction == "sum" else numerator / counts.sum()
    if not return_details:
        return loss
    return loss, {"per_sample_bc_sum": bc_samples.detach(), "per_sample_valid_coordinates": counts.detach(),
                  "bc_sum": bc_samples.detach().sum(), "moe_balance_sum": balance.detach().sum(),
                  "moe_z_sum": z.detach().sum(), "router_loads": loads}
