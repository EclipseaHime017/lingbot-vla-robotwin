"""Episode-stratified validation with sample-indexed flow noise on every rank."""
from __future__ import annotations

import hashlib
import math
from collections import defaultdict

import torch
import torch.distributed as dist

from .training_data import model_collate
from .training_identity import canonical_sha256


def validation_samples(dataset, *, seed=12345, frames_per_episode=1):
    if frames_per_episode <= 0:
        raise ValueError("validation_frames_per_episode must be positive.")
    samples, offset = [], 0
    for child in dataset.datasets:
        groups = defaultdict(list)
        episode_ids = child.dataset.hf_dataset["episode_index"]
        indices = getattr(child, "sample_indices", None)
        for local, raw_index in enumerate(indices if indices is not None else range(len(child))):
            groups[int(episode_ids[raw_index])].append(local)
        for episode, indices in sorted(groups.items()):
            digest = hashlib.sha256(f"{seed}:{child.task}:{episode}".encode()).digest()
            rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
            chosen = torch.randperm(len(indices), generator=rng)[:frames_per_episode].tolist()
            samples.extend({"index": offset + indices[i], "task": child.task, "episode": episode} for i in chosen)
        offset += len(child)
    return samples


def validation_identity(samples, *, seed=12345, frames_per_episode=1):
    return {"selection": "every_episode_v1", "noise": "sample_index_cpu_beta_v1", "seed": seed,
            "frames_per_episode": frames_per_episode, "frames": len(samples),
            "tasks": len({row["task"] for row in samples}),
            "episodes": len({(row["task"], row["episode"]) for row in samples}),
            "selection_sha256": canonical_sha256(samples)}


def validation_draws(indices, action_shape, *, seed=12345):
    noise, times = [], []
    for index in indices:
        digest = hashlib.sha256(f"validation-fm-v1:{seed}:{index}".encode()).digest()
        rng = torch.Generator().manual_seed(int.from_bytes(digest[:8], "little") % (2**63))
        noise.append(torch.randn(action_shape, generator=rng, dtype=torch.float32))
        times.append(torch.rand((), generator=rng).pow(2 / 3) * 0.999 + 0.001)
    return torch.stack(noise), torch.stack(times)


def validate_stratified(model, dataset, device, *, batch_size, samples, loss_fn=None, rank=0, world_size=1,
                        frames_per_episode=1):
    if not samples:
        return {"validation_bc_loss": None, "validation_per_task": {}, "validation_coverage": validation_identity(samples)}
    device = torch.device(device)
    tasks = sorted({row["task"] for row in samples})
    task_ids = {task: i for i, task in enumerate(tasks)}
    totals = torch.zeros((len(tasks), 2), dtype=torch.float64, device=device)
    was_training = model.training
    model.eval()
    gpu_ids = [device.index if device.index is not None else torch.cuda.current_device()] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=gpu_ids), torch.no_grad():
            for step in range(math.ceil(len(samples) / (batch_size * world_size))):
                positions = [step * batch_size * world_size + rank * batch_size + i for i in range(batch_size)]
                rows = [samples[min(i, len(samples) - 1)] for i in positions]
                batch = model_collate([dataset[row["index"]] for row in rows])
                noise, time = validation_draws([row["index"] for row in rows], batch["actions"].shape[1:])
                batch = {key: value.to(device) for key, value in batch.items()}
                with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
                    kwargs = dict(noise=noise, time=time, auxiliary_scale=0, return_details=True)
                    _, details = loss_fn(model, batch, **kwargs) if loss_fn else model(batch, **kwargs)
                for i, (position, row) in enumerate(zip(positions, rows)):
                    if position < len(samples):
                        totals[task_ids[row["task"]], 0] += details["per_sample_bc_sum"][i].double()
                        totals[task_ids[row["task"]], 1] += details["per_sample_valid_coordinates"][i].double()
        if world_size > 1:
            dist.all_reduce(totals)
    finally:
        model.train(was_training)
    total = totals.sum(0)
    return {"validation_bc_loss": float(total[0] / total[1]),
            "validation_per_task": {task: float(totals[i, 0] / totals[i, 1]) for task, i in task_ids.items()},
            "validation_coverage": validation_identity(samples, frames_per_episode=frames_per_episode)}
