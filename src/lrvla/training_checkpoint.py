"""Stream official weights and export merged, strict-loadable hf_ckpt folders."""
from __future__ import annotations

import json
from pathlib import Path
import random

import numpy as np
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from .training_models import LoRALinear
from .training_identity import base_checkpoint_identity


def run_relative_path(run, value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (Path(run) / path).resolve()


def load_safetensors_streamed(model, checkpoint, dtype=torch.bfloat16, device="cpu"):
    files = sorted(Path(checkpoint).glob("*.safetensors"))
    if not files:
        raise FileNotFoundError(f"No safetensors shards in {checkpoint}; pass the nested hf_ckpt directory.")
    expected = model.state_dict()
    # Verify completeness and geometry from headers before allocating GPU weights.
    inventory = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name not in expected:
                    raise ValueError(f"Unexpected checkpoint key: {name}")
                if name in inventory:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                shape = tuple(reader.get_slice(name).get_shape())
                if shape != tuple(expected[name].shape):
                    raise ValueError(f"Shape mismatch for {name}: checkpoint {shape}, model {tuple(expected[name].shape)}")
                inventory.add(name)
    missing = set(expected) - inventory
    if missing:
        raise ValueError(f"Incomplete checkpoint: missing {len(missing)} tensors, e.g. {sorted(missing)[:8]}. Finish every shard download and preserve the base architecture.")
    seen = set()
    for path in files:
        with safe_open(path, framework="pt", device="cpu") as reader:
            keys = reader.keys()
            unknown = set(keys) - set(expected)
            if unknown:
                raise ValueError(f"Unexpected checkpoint keys: {sorted(unknown)[:8]}")
            for name in keys:
                if name in seen:
                    raise ValueError(f"Duplicate checkpoint tensor: {name}")
                if tuple(reader.get_slice(name).get_shape()) != tuple(expected[name].shape):
                    raise ValueError(f"Shape mismatch for {name}: checkpoint {reader.get_slice(name).get_shape()}, model {expected[name].shape}")
                value = reader.get_tensor(name)
                if value.is_floating_point():
                    value = value.to(device=device, dtype=dtype)
                else:
                    value = value.to(device=device)
                # Assign one tensor at a time, avoiding a full-model host copy.
                parent_path, _, leaf = name.rpartition(".")
                parent = model.get_submodule(parent_path) if parent_path else model
                if leaf in parent._parameters:
                    old = parent._parameters[leaf]
                    parent._parameters[leaf] = torch.nn.Parameter(value, requires_grad=old.requires_grad)
                elif leaf in parent._buffers:
                    parent._buffers[leaf] = value
                else:
                    raise KeyError(f"Checkpoint tensor {name} has no corresponding module slot")
                seen.add(name)
    missing = set(expected) - seen
    if missing:
        raise ValueError(f"Missing checkpoint tensors: {sorted(missing)[:12]}. Preserve the base checkpoint architecture/config.")
    if any(p.is_meta for p in model.parameters()):
        raise AssertionError("Unloaded meta parameters remain.")
    return {"shards": len(files), "tensors": len(seen)}


def merged_state_items(model, dtype=torch.bfloat16):
    adapters = {name: module for name, module in model.named_modules() if isinstance(module, LoRALinear)}
    for name, tensor in model.state_dict().items():
        adapter_name, _, suffix = name.rpartition(".")
        if adapter_name in adapters and suffix in {"lora_A", "lora_B"}:
            continue
        if ".base." in name:
            prefix, tail = name.rsplit(".base.", 1)
            if prefix in adapters:
                value = adapters[prefix].merged_weight() if tail == "weight" else tensor.detach().cpu()
                name = f"{prefix}.{tail}"
            else:
                value = tensor.detach().cpu()
        else:
            value = tensor.detach().cpu()
        if value.is_floating_point():
            value = value.to(dtype)
        yield name, value.contiguous()


def base_inventory(checkpoint):
    inventory = {}
    for path in sorted(Path(checkpoint).glob("*.safetensors")):
        with safe_open(path, framework="pt", device="cpu") as reader:
            for name in reader.keys():
                if name in inventory:
                    raise ValueError(f"Duplicate base tensor: {name}")
                view = reader.get_slice(name)
                inventory[name] = (path, tuple(view.get_shape()), view.get_dtype())
    if not inventory:
        raise FileNotFoundError(f"No base safetensors in {checkpoint}")
    return inventory


def restore_trainable_base(model, checkpoint):
    """Initialize FP32 masters from original base, before optimizer/resume.

    Frozen weights keep their memory-saving BF16 training representation. New
    LoRA matrices retain their initialized values; their frozen base is restored
    from the original checkpoint only when exporting.
    """
    parameters = {name: p for name, p in model.named_parameters() if p.requires_grad}
    inventory = base_inventory(checkpoint)
    with torch.no_grad():
        for name, parameter in parameters.items():
            if name.endswith((".lora_A", ".lora_B")):
                continue
            if name not in inventory or inventory[name][1] != tuple(parameter.shape):
                raise ValueError(f"Trainable parameter missing or mismatched in base: {name}")
            with safe_open(inventory[name][0], framework="pt", device="cpu") as reader:
                parameter.copy_(reader.get_tensor(name).float())


def overlay_base_state_items(checkpoint, trainable_state, selection, *, expected_base_identity):
    """Original frozen tensors plus FP32 masters; never upcast a rounded base."""
    if base_checkpoint_identity(checkpoint) != expected_base_identity:
        raise ValueError("Export base_identity differs from the training checkpoint.")
    inventory = base_inventory(checkpoint)
    if set(trainable_state) != set(selection["trainable_names"]):
        raise ValueError("Export trainable parameter keys differ from the saved selection.")
    adapters = set(selection.get("lora_modules", []))
    rank = (selection.get("lora") or {}).get("rank", 0)
    alpha = (selection.get("lora") or {}).get("alpha", 0)
    for name, value in trainable_state.items():
        if value.dtype != torch.float32:
            raise ValueError(f"Export requires FP32 master weights: {name} is {value.dtype}")
        parent, _, leaf = name.rpartition(".")
        if parent in adapters and leaf in {"lora_A", "lora_B"}:
            if rank <= 0 or f"{parent}.weight" not in inventory:
                raise ValueError(f"Invalid LoRA base/rank: {parent}")
            shape = inventory[f"{parent}.weight"][1]
            expected = (rank, shape[1]) if leaf == "lora_A" else (shape[0], rank)
        else:
            if name not in inventory:
                raise ValueError(f"Trainable key absent from original base: {name}")
            expected = inventory[name][1]
        if tuple(value.shape) != expected:
            raise ValueError(f"Export shape mismatch: {name}")
    for parent in adapters:
        if not {f"{parent}.lora_A", f"{parent}.lora_B"} <= trainable_state.keys():
            raise ValueError(f"Incomplete LoRA pair: {parent}")
    for name, (_, _, source_dtype) in inventory.items():
        if name not in trainable_state and source_dtype in {"BF16", "F16", "F64"}:
            raise ValueError(f"Original frozen base must contain F32 weights: {name} is {source_dtype}")
    # Validate before returning the iterator, so invalid inputs create no export.
    def items():
        for name, (path, _, source_dtype) in inventory.items():
            if name in trainable_state:
                value = trainable_state[name].detach().cpu()
            else:
                with safe_open(path, framework="pt", device="cpu") as reader:
                    value = reader.get_tensor(name)
                if value.is_floating_point() and source_dtype != "F32":
                    raise ValueError(f"Original frozen base must contain F32 weights: {name} is {source_dtype}")
                parent, _, leaf = name.rpartition(".")
                if parent in adapters and leaf == "weight":
                    with torch.autocast("cpu", enabled=False):
                        value = value.float() + (trainable_state[f"{parent}.lora_B"].detach().float().cpu() @
                                                 trainable_state[f"{parent}.lora_A"].detach().float().cpu()) * (alpha / rank)
            yield name, value.contiguous()
    return items()


def export_hf_checkpoint(model, output, *, dtype=torch.bfloat16, max_shard_bytes=512 * 2**20, processor=None,
                         state_items=None, base_checkpoint=None, expected_base_identity=None, selection=None):
    provenance = {}
    if base_checkpoint is not None:
        if state_items is not None:
            raise ValueError("Specify base overlay or state_items, not both.")
        masters = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
        state_items = overlay_base_state_items(base_checkpoint, masters, selection,
                                              expected_base_identity=expected_base_identity)
        provenance = {"frozen_source": "original_base", "trainable_source": "fp32_master",
                      "base_identity": expected_base_identity}
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to mix an export with existing files: {output}")
    output.mkdir(parents=True, exist_ok=True)
    index, total, shard, shard_bytes, shards = {}, 0, {}, 0, []

    def flush():
        nonlocal shard, shard_bytes
        if not shard:
            return
        filename = f"model-{len(shards) + 1:05d}.safetensors"
        save_file(shard, str(output / filename), metadata={"format": "pt"})
        for name in shard:
            index[name] = filename
        shards.append(filename)
        shard, shard_bytes = {}, 0

    items = merged_state_items(model, dtype) if state_items is None else state_items
    for name, value in items:
        value = value.detach().cpu()
        if value.is_floating_point():
            value = value.to(dtype)
        value = value.contiguous()
        size = value.numel() * value.element_size()
        if shard and shard_bytes + size > max_shard_bytes:
            flush()
        shard[name] = value
        shard_bytes += size
        total += size
    flush()
    (output / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total}, "weight_map": index}, indent=2))
    if hasattr(model.config, "to_json_file"):
        model.config.to_json_file(str(output / "config.json"))
        config_path = output / "config.json"
        saved_config = json.loads(config_path.read_text())
        saved_config["torch_dtype"] = str(dtype).split(".")[-1]
        config_path.write_text(json.dumps(saved_config, indent=2) + "\n")
    if processor is not None:
        processor.save_pretrained(output)
    model_type = getattr(model.config, "model_type", "")
    export_format = "synthetic_smoke" if model_type == "synthetic_smoke" else "official_lingbot_vla_v2"
    adapter_count = sum(isinstance(module, LoRALinear) for module in model.modules())
    (output / "export.json").write_text(json.dumps({"format": export_format, "merged_lora": bool(adapter_count),
                                                   "lora_modules": adapter_count, "dtype": str(dtype),
                                                   "tensors": len(index), "shards": len(shards), **provenance}, indent=2))
    return output


def save_training_checkpoint(model, optimizer, scheduler, output, state, metadata):
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    weights = {name: p.detach().cpu() for name, p in model.named_parameters() if p.requires_grad}
    numpy_rng = np.random.get_state()
    payload = {"trainable_state": weights, "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
               "state": state, "metadata": metadata, "torch_rng": torch.get_rng_state(),
               "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
               "python_rng": random.getstate(),
               "numpy_rng": (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:])}
    temporary = output.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(output)


def resume_training_checkpoint(model, optimizer, scheduler, checkpoint, *, expected_metadata):
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    # A run may change max_steps or checkpoint intervals; parameter/data identity must match.
    for key in ("base_identity", "manifest_content_sha256"):
        if key not in payload["metadata"] or key not in expected_metadata:
            raise ValueError(f"Checkpoint lacks content identity {key}; cannot verify the frozen base and clean data for portable resume.")
    for key in ("base_identity", "manifest_content_sha256", "selection", "norm_sha256", "model_geometry", "qwen_config_sha256"):
        if payload["metadata"].get(key) != expected_metadata.get(key):
            raise ValueError(f"Resume mismatch for {key}; use identical frozen base content, audited clean data and trainable selection.")
    for key in ("loss_normalization", "moe_loss", "data_sampling"):
        if payload["metadata"].get(key) != expected_metadata.get(key):
            raise ValueError(f"Resume mismatch for {key}; use the same loss normalization and data sampling. "
                             "Checkpoints predating the tail/normalization fix cannot provide an exact resume.")
    parameters = dict(model.named_parameters())
    trainable = {name for name, p in parameters.items() if p.requires_grad}
    if set(payload["trainable_state"]) != trainable:
        raise ValueError("Resume trainable parameter keys differ from the current model.")
    with torch.no_grad():
        for name, value in payload["trainable_state"].items():
            parameters[name].copy_(value)
    optimizer.load_state_dict(payload["optimizer"])
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    numpy_rng = payload["numpy_rng"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    if payload["cuda_rng"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    return payload["state"]
