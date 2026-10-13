"""Collective trainable-only FSDP checkpoints and complete inference exports."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.checkpoint.state_dict import (
    StateDictOptions, get_model_state_dict, get_state_dict, set_state_dict,
)

from .training_checkpoint import export_hf_checkpoint


FORMAT = "lrvla-fsdp-v1"
IDENTITY_KEYS = (
    "base_identity", "manifest_content_sha256", "selection", "norm_sha256",
    "model_geometry", "qwen_config_sha256", "loss_normalization", "data_sampling", "distributed",
)


def _check_group(rank, world_size):
    if not dist.is_initialized():
        raise RuntimeError("Distributed checkpoints require an initialized process group.")
    if rank != dist.get_rank() or world_size != dist.get_world_size():
        raise ValueError("Checkpoint rank/world_size differs from the active process group.")


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def _rank_path(directory, rank):
    return directory / f"rank-{rank:05d}.pt"


def _capture_rank_state(scheduler, rank, world_size, device):
    numpy_rng = np.random.get_state()
    device = torch.device(device)
    return {
        "format": FORMAT, "rank": rank, "world_size": world_size,
        "scheduler": scheduler.state_dict(), "torch_rng": torch.get_rng_state(),
        # Accessing every device's RNG would create contexts on other ranks'
        # GPUs. Each process saves only the device it actually owns.
        "cuda_rng": torch.cuda.get_rng_state(device) if device.type == "cuda" else None,
        "python_rng": random.getstate(),
        "numpy_rng": (numpy_rng[0], torch.from_numpy(numpy_rng[1].astype(np.int64)), *numpy_rng[2:]),
    }


def _restore_rank_state(payload, scheduler, device):
    scheduler.load_state_dict(payload["scheduler"])
    torch.set_rng_state(payload["torch_rng"])
    random.setstate(payload["python_rng"])
    numpy_rng = payload["numpy_rng"]
    np.random.set_state((numpy_rng[0], numpy_rng[1].numpy().astype(np.uint32), *numpy_rng[2:]))
    if payload["cuda_rng"] is not None:
        device = torch.device(device)
        if device.type != "cuda":
            raise ValueError("A CUDA checkpoint requires a CUDA device to restore its RNG.")
        torch.cuda.set_rng_state(payload["cuda_rng"], device)


def _checkpoint_options():
    return StateDictOptions(ignore_frozen_params=True, cpu_offload=True, strict=False)


def save_distributed_checkpoint(model, optimizer, scheduler, output, state, metadata,
                                *, rank, world_size, device):
    """All ranks save model/Adam shards and their own exact continuation state."""
    _check_group(rank, world_size)
    output = Path(output)
    if (output / "complete.json").exists():
        raise FileExistsError(f"Refusing to overwrite a complete distributed checkpoint: {output}")
    output.mkdir(parents=True, exist_ok=True)
    rank_state = _capture_rank_state(scheduler, rank, world_size, device)
    model_state, optim_state = get_state_dict(model, optimizer, options=_checkpoint_options())
    dcp.save({"model": model_state, "optimizer": optim_state}, checkpoint_id=output / "shards")
    rank_path = _rank_path(output, rank)
    temporary = rank_path.with_suffix(".tmp")
    torch.save(rank_state, temporary)
    temporary.replace(rank_path)
    if rank == 0:
        _atomic_json(output / "run_state.json", {
            "format": FORMAT, "world_size": world_size, "state": state, "metadata": metadata,
        })
    dist.barrier()
    if rank == 0:
        for index in range(world_size):
            if not _rank_path(output, index).is_file():
                raise RuntimeError(f"Checkpoint missing continuation state for rank {index}.")
        _atomic_json(output / "complete.json", {
            "format": FORMAT, "world_size": world_size,
            "run_state_sha256": hashlib.sha256((output / "run_state.json").read_bytes()).hexdigest(),
        })
    dist.barrier()
    return output


def _read_checkpoint(checkpoint, expected_metadata, rank, world_size):
    complete_path = checkpoint / "complete.json"
    if not complete_path.is_file():
        raise ValueError(f"Distributed checkpoint is incomplete: {checkpoint}")
    complete = json.loads(complete_path.read_text())
    run_path = checkpoint / "run_state.json"
    if not run_path.is_file():
        raise ValueError("Distributed checkpoint lacks run_state.json.")
    if complete.get("run_state_sha256") != hashlib.sha256(run_path.read_bytes()).hexdigest():
        raise ValueError("Distributed checkpoint run_state.json does not match its completion marker.")
    run = json.loads(run_path.read_text())
    if complete.get("format") != FORMAT or run.get("format") != FORMAT:
        raise ValueError("Unsupported distributed checkpoint format.")
    if complete.get("world_size") != world_size or run.get("world_size") != world_size:
        raise ValueError("Resume world_size differs; exact distributed data/RNG continuation requires the same GPU count.")
    saved_metadata = run["metadata"]
    # JSON sidecars preserve semantic values but canonicalize tuples to lists.
    expected_metadata = json.loads(json.dumps(expected_metadata))
    for key in ("base_identity", "manifest_content_sha256"):
        if key not in saved_metadata or key not in expected_metadata:
            raise ValueError(f"Checkpoint lacks content identity {key}.")
    for key in IDENTITY_KEYS:
        if saved_metadata.get(key) != expected_metadata.get(key):
            raise ValueError(f"Resume mismatch for {key}; use the same base, clean data, model selection and sampling.")
    if not (checkpoint / "shards" / ".metadata").is_file():
        raise ValueError("Distributed checkpoint lacks the DCP shard metadata.")
    rank_file = _rank_path(checkpoint, rank)
    if not rank_file.is_file():
        raise ValueError(f"Distributed checkpoint lacks continuation state for rank {rank}.")
    rank_state = torch.load(rank_file, map_location="cpu", weights_only=True)
    if (rank_state.get("format") != FORMAT or rank_state.get("rank") != rank
            or rank_state.get("world_size") != world_size):
        raise ValueError("Checkpoint rank continuation state differs from the active rank/world_size.")
    return run, rank_state


def resume_distributed_checkpoint(model, optimizer, scheduler, checkpoint, *, expected_metadata,
                                  rank, world_size, device):
    """Restore shards, scheduler and this rank's RNG after identity validation."""
    _check_group(rank, world_size)
    checkpoint = Path(checkpoint)
    run, rank_state = _read_checkpoint(checkpoint, expected_metadata, rank, world_size)
    options = _checkpoint_options()
    # PyTorch's get_state_dict initializes an empty Adam state using a zero-lr
    # step, preserving parameters while supplying correctly sharded load slots.
    model_state, optim_state = get_state_dict(model, optimizer, options=options)
    payload = {"model": model_state, "optimizer": optim_state}
    dcp.load(payload, checkpoint_id=checkpoint / "shards")
    set_state_dict(model, optimizer, model_state_dict=payload["model"],
                   optim_state_dict=payload["optimizer"], options=options)
    _restore_rank_state(rank_state, scheduler, device)
    return run["state"]


def export_distributed_hf(model, policy, output, *, dtype, processor, rank):
    """All ranks gather the full VLA; rank zero writes inference-ready shards."""
    _check_group(rank, dist.get_world_size() if dist.is_initialized() else 0)
    state = get_model_state_dict(model, options=StateDictOptions(full_state_dict=True, cpu_offload=True))
    exported = None
    if rank == 0:
        def policy_state_items():
            for name, value in state.items():
                if not name.startswith("policy."):
                    raise ValueError(f"Unexpected state key outside BCForward.policy: {name}")
                yield name.removeprefix("policy."), value
        exported = export_hf_checkpoint(policy, output, dtype=dtype, processor=processor,
                                        state_items=policy_state_items())
    dist.barrier()
    return exported
