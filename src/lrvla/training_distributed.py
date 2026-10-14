"""Single-node FSDP training of the complete Action Expert with a frozen VLM."""
from __future__ import annotations

from datetime import timedelta
import hashlib
import json
import math
import os
from pathlib import Path
import random
import shutil
import sys
import time

import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Sampler
import yaml

from .training_data import build_clean_datasets, model_collate
from .training_fsdp import fsdp_shard_report, wrap_expert_fsdp
from .training_identity import base_checkpoint_identity, manifest_content_sha256
from .training_models import configure_trainable, memory_report, normalize_bc_gradients
from .training_runtime import build_policy, enable_action_checkpointing


def accumulation_for_world(config, world_size):
    """Preserve an explicit global batch while changing the number of GPUs."""
    batch = int(config.get("batch_size", 1))
    if world_size <= 0 or batch <= 0:
        raise ValueError("world_size and batch_size must be positive.")
    global_batch = config.get("global_batch_size")
    if global_batch is None:
        accumulation = int(config.get("gradient_accumulation", 16))
    else:
        global_batch = int(global_batch)
        if global_batch <= 0 or global_batch % (batch * world_size):
            raise ValueError("global_batch_size must be a positive multiple of batch_size * world_size.")
        accumulation = global_batch // (batch * world_size)
    if accumulation <= 0:
        raise ValueError("gradient_accumulation must be positive.")
    if config.get("require_no_gradient_accumulation", False) and accumulation != 1:
        raise ValueError("This configuration forbids gradient accumulation; set batch_size * world_size = global_batch_size.")
    return accumulation, batch * world_size * accumulation


class DistributedEpochBatchSampler(Sampler):
    """Shard one deterministic global permutation and replay consumed batches.

    The final global batch is padded from the start of the same permutation so
    every rank executes the same number of forward and backward collectives.
    """

    def __init__(self, length, batch_size, seed, *, rank, world_size, epoch=0, offset=0):
        if length <= 0 or batch_size <= 0 or world_size <= 0 or not 0 <= rank < world_size or offset < 0:
            raise ValueError("Invalid distributed sampler dimensions or offset.")
        self.length, self.batch_size, self.seed = length, batch_size, seed
        self.rank, self.world_size, self.epoch, self.offset = rank, world_size, epoch, offset

    def __iter__(self):
        order = torch.randperm(self.length, generator=torch.Generator().manual_seed(self.seed + self.epoch)).tolist()
        width = self.batch_size * self.world_size
        padding = (-len(order)) % width
        if padding:
            order += (order * math.ceil(padding / self.length))[:padding]
        for start in range(self.offset * width, len(order), width):
            rank_start = start + self.rank * self.batch_size
            yield order[rank_start:rank_start + self.batch_size]

    def __len__(self):
        return max(0, math.ceil(self.length / (self.batch_size * self.world_size)) - self.offset)


def distributed_dataloader(dataset, batch_size, seed, *, rank, world_size, epoch=0, offset=0, num_workers=0):
    sampler = DistributedEpochBatchSampler(len(dataset), batch_size, seed, rank=rank,
                                          world_size=world_size, epoch=epoch, offset=offset)
    rng = torch.Generator().manual_seed(seed + 100000 + epoch + rank)
    return DataLoader(dataset, batch_sampler=sampler, collate_fn=model_collate,
                      num_workers=num_workers, pin_memory=True, generator=rng)


def coordinate_denominator(global_valid_count, world_size):
    """FSDP averages rank gradients; recover the global coordinate mean."""
    if not math.isfinite(global_valid_count) or global_valid_count <= 0 or world_size <= 0:
        raise ValueError("The global valid action count and world size must be positive and finite.")
    return global_valid_count / world_size


def all_ranks_finite(value, device):
    flag = torch.tensor(int(bool(torch.isfinite(value).all())), device=device, dtype=torch.int32)
    dist.all_reduce(flag, op=dist.ReduceOp.MIN)
    return bool(flag.item())


def validate_distributed(model, dataset, device, *, batch_size, max_batches, rank, world_size):
    """Evaluate the same first max_batches global batches on any GPU count."""
    limit = min(len(dataset), max_batches * batch_size)
    if not limit:
        return None
    rounds = math.ceil(limit / (batch_size * world_size))
    totals = torch.zeros(2, dtype=torch.float64, device=device)
    model.eval()
    with torch.random.fork_rng(devices=[device.index]), torch.no_grad():
        torch.manual_seed(12345 + rank)
        for step in range(rounds):
            ids = [step * batch_size * world_size + rank * batch_size + i for i in range(batch_size)]
            batch = model_collate([dataset[min(index, limit - 1)] for index in ids])
            # Mark padded samples only for aggregation; each rank must still
            # participate in the same FSDP forwards, including a final round.
            real_mask = torch.tensor([index < limit for index in ids], dtype=torch.bool)
            batch["joint_mask"][~real_mask] = False
            valid = float(batch["joint_mask"].float().sum())
            if not valid:
                # masked_bc_loss rejects an entirely padded batch. Compute a
                # harmless duplicate and give it zero weight below.
                batch = model_collate([dataset[limit - 1] for _ in ids])
            batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                numerator = model(batch, reduction="sum")
            if valid:
                totals[0] += numerator.double()
                totals[1] += valid
    dist.all_reduce(totals)
    model.train()
    return float(totals[0] / totals[1]) if float(totals[1]) else None


def sharded_memory_estimate(report, world_size, *, largest_shard_parameters=None):
    trainable = report["trainable_parameters"]
    frozen = report["total_parameters"] - trainable
    estimate = {
        "persistent_gib_per_gpu": (frozen * 2 + trainable * 16 / world_size) / 2**30,
        "root_fsdp_temporary_upper_gib_per_gpu": (frozen * 2 + trainable * 16 / world_size + trainable * 8) / 2**30,
        "excludes": "activations, frozen VLM KV cache, allocator reserve, CUDA/NCCL context",
        "frozen_vlm_replicated": True,
        "trainable_parameters_sharded": True,
    }
    if largest_shard_parameters is not None:
        estimate.update(
            largest_shard_parameters=largest_shard_parameters,
            layer_fsdp_temporary_budget_gib_per_gpu=(frozen * 2 + trainable * 16 / world_size +
                                                     largest_shard_parameters * 16) / 2**30,
        )
    return estimate


def train_distributed(config):
    from .training import PROJECT_ROOT, portable_inference_cli, prepare_runtime_files, resolve_path, schedule_multiplier
    from .training_distributed_checkpoint import (
        export_distributed_hf, resume_distributed_checkpoint, save_distributed_checkpoint,
    )

    config = dict(config)
    fsdp_wrap = config.get("fsdp_wrap", "root")
    if fsdp_wrap not in {"root", "decoder_layer"}:
        raise ValueError("fsdp_wrap must be root or decoder_layer.")
    layer_sharding = fsdp_wrap == "decoder_layer"
    world_size, rank, local_rank = (int(os.environ[name]) for name in ("WORLD_SIZE", "RANK", "LOCAL_RANK"))
    if config.get("mode") != "expert" or int(config.get("expert_last_n_layers", 2)) != -1:
        raise ValueError("Distributed training currently supports the full Action Expert: mode=expert, expert_last_n_layers=-1.")
    if not torch.cuda.is_available() or torch.cuda.device_count() < world_size:
        raise RuntimeError("The distributed entrypoint needs all PBS-assigned GPUs on one node.")
    if config.get("action_chunk_tail", "mask") not in {"mask", "drop"}:
        raise ValueError("action_chunk_tail must be mask or drop.")
    if not 0 <= int(config.get("val_episodes", 5)) < 50:
        raise ValueError("val_episodes must be between 0 and 49.")
    for key in ("max_steps", "save_every", "eval_every"):
        if int(config.get(key, 1000 if key == "max_steps" else 100)) <= 0:
            raise ValueError(f"{key} must be positive.")
    accumulation, global_batch = accumulation_for_world(config, world_size)
    config.update(gradient_accumulation=accumulation, global_batch_size=global_batch,
                  distributed={"backend": "nccl", "strategy": "fsdp_full_shard", "world_size": world_size,
                               "wrap_policy": fsdp_wrap})
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    dist.init_process_group("nccl", timeout=timedelta(minutes=30), device_id=device)
    try:
        seed = int(config.get("seed", 42))
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        base, qwen = resolve_path(config["base_checkpoint"]), resolve_path(config["qwen_path"])
        if "robotwin" in base.name.lower():
            raise ValueError("Use the original VLA base checkpoint.")
        architecture = yaml.safe_load(resolve_path(config["architecture_yaml"]).read_text())
        architecture["train"]["chunk_size"] = int(config.get("chunk_size", 50))
        manifest_path, norm_path = resolve_path(config["manifest"]), resolve_path(config["norm_stats"])
        from .data import load_manifest
        manifest = load_manifest(manifest_path)
        if not manifest.get("complete_50_tasks") and not config.get("allow_subset", False):
            raise ValueError("Training requires all 50 clean tasks; --allow-subset is for debugging only.")
        output = resolve_path(config["output_dir"])
        shared = [None]
        if rank == 0:
            data_config, robot_config, manifest_digest = prepare_runtime_files(
                config, architecture, manifest_path, norm_path, output)
            print("Verifying the base checkpoint SHA-256 identity...", flush=True)
            shared[0] = (data_config, str(robot_config), manifest_digest, base_checkpoint_identity(base))
        dist.broadcast_object_list(shared, src=0, device=device)
        data_config, robot_config, manifest_digest, base_identity = shared[0]
        policy = build_policy(PROJECT_ROOT, architecture, base, qwen, device=device)
        selection = configure_trainable(policy, config)
        report = memory_report(policy)
        shard_report = fsdp_shard_report(policy, layer_sharding=layer_sharding)
        estimate = sharded_memory_estimate(
            report, world_size,
            largest_shard_parameters=shard_report["largest_shard_parameters"] if layer_sharding else None)
        total_gib = torch.cuda.get_device_properties(device).total_memory / 2**30
        if estimate["persistent_gib_per_gpu"] > total_gib * 0.92:
            raise RuntimeError("Sharded persistent parameter/optimizer state exceeds available GPU memory.")
        from transformers import AutoProcessor
        processor = AutoProcessor.from_pretrained(qwen, padding_side="right", local_files_only=True)
        datasets, validation, splits = build_clean_datasets(
            manifest, data_config, policy.config, processor, Path(robot_config), int(config.get("val_episodes", 5)),
            action_chunk_tail=config.get("action_chunk_tail", "mask"))
        if config.get("gradient_checkpointing", True):
            enable_action_checkpointing(policy)
        train_values = {**architecture["train"], **policy.config.to_dict()}
        train_values.update(output_dir=str(output), train_expert_only=True, freeze_vision_encoder=True,
                            attention_implementation="eager", vit_attn_implementation="sdpa", use_compile=False)
        cli = {"model": {**architecture["model"], "model_path": str(base), "tokenizer_path": str(qwen)},
               "train": train_values, "data": data_config}
        metadata = {
            "base_checkpoint": str(base), "base_identity": base_identity,
            "manifest_sha256": manifest_digest, "manifest_content_sha256": manifest_content_sha256(manifest),
            "selection": selection, "training_method": "clean_only_behavior_cloning",
            "competition_all_50_tasks": bool(manifest.get("complete_50_tasks")),
            "loss_normalization": "valid_action_coordinate_mean_v1",
            "data_sampling": {"action_chunk_tail": config.get("action_chunk_tail", "mask"),
                              "validation_action_chunk_tail": "mask", "val_episodes": int(config.get("val_episodes", 5)),
                              "seed": seed, "batch_size": int(config.get("batch_size", 1)),
                              "gradient_accumulation": accumulation, "world_size": world_size,
                              "global_batch_size": global_batch, "epoch_end_padding": "repeat_initial_permutation_samples"},
            "distributed": config["distributed"], "memory": report, "sharded_memory_estimate": estimate,
            "fsdp_shards": shard_report,
            "training_samples": len(datasets), "validation_samples": len(validation) if validation else 0,
            "norm_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
            "qwen_config_sha256": hashlib.sha256((qwen / "config.json").read_bytes()).hexdigest(),
            "model_geometry": {"chunk_size": policy.config.chunk_size, "max_action_dim": policy.config.max_action_dim,
                               "max_state_dim": policy.config.max_state_dim, "img_size": data_config["img_size"]},
        }
        if rank == 0:
            qwen_assets = output / "assets" / "qwen3_vl"
            processor.save_pretrained(qwen_assets)
            shutil.copy2(qwen / "config.json", qwen_assets / "config.json")
            saved_cli = portable_inference_cli(cli)
            saved_cli["train"]["tokenizer_path"] = "assets/qwen3_vl"
            (output / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(saved_cli, sort_keys=False))
            (output / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
            (output / "episode_splits.json").write_text(json.dumps(splits, indent=2))
            (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
            print(json.dumps({"world_size": world_size, "global_batch_size": global_batch,
                              "per_rank_batch_size": config.get("batch_size", 1), "gradient_accumulation": accumulation,
                              "training_samples": len(datasets), "memory": report,
                              "sharded_memory_estimate": estimate}), flush=True)
        model = wrap_expert_fsdp(policy, device=device, layer_sharding=layer_sharding)
        params = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(params, lr=float(config.get("lr", 5e-5)),
                                      weight_decay=float(config.get("weight_decay", 0.01)), foreach=False)
        max_steps = int(config.get("max_steps", 1000))
        warmup = min(int(config.get("warmup_steps", 100)), max_steps)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_multiplier(step, max_steps, warmup))
        # Model construction is identical across ranks; FM noise is independent.
        torch.manual_seed(seed + rank)
        np.random.seed(seed + rank)
        random.seed(seed + rank)
        state = {"global_step": 0, "epoch": 0, "batch_offset": 0}
        if config.get("resume"):
            state = resume_distributed_checkpoint(
                model, optimizer, scheduler, resolve_path(config["resume"]), expected_metadata=metadata,
                rank=rank, world_size=world_size, device=device)
        batch_size, workers = int(config.get("batch_size", 1)), int(config.get("num_workers", 0))

        def iterator():
            return iter(distributed_dataloader(datasets, batch_size, seed, rank=rank, world_size=world_size,
                                              epoch=state["epoch"], offset=state["batch_offset"], num_workers=workers))

        batches = iterator()
        model.train()
        optimizer.zero_grad(set_to_none=True)
        while state["global_step"] < max_steps:
            torch.cuda.synchronize(device)
            start = time.monotonic()
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            for _ in range(accumulation):
                try:
                    batch = next(batches)
                except StopIteration:
                    state["epoch"] += 1
                    state["batch_offset"] = 0
                    batches = iterator()
                    batch = next(batches)
                state["batch_offset"] += 1
                batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    loss = model(batch, reduction="sum")
                if not all_ranks_finite(loss, device):
                    raise FloatingPointError("A rank produced a nonfinite BC loss.")
                # Reduce-scatter on every microbatch keeps accumulated gradients
                # sharded. no_sync would retain the entire Expert gradient.
                loss.backward()
                totals[0] += loss.detach().double()
                totals[1] += batch["joint_mask"].double().sum()
            dist.all_reduce(totals)
            valid_count = float(totals[1])
            normalize_bc_gradients(params, coordinate_denominator(valid_count, world_size))
            grad = model.clip_grad_norm_(float(config.get("max_grad_norm", 1)))
            if not all_ranks_finite(grad, device):
                raise FloatingPointError("A rank produced a nonfinite gradient norm.")
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            update_seconds = time.monotonic() - start
            state["global_step"] += 1
            step = state["global_step"]
            memory = torch.tensor([torch.cuda.max_memory_allocated(device) / 2**30,
                                   torch.cuda.max_memory_reserved(device) / 2**30], device=device)
            memories = [torch.empty_like(memory) for _ in range(world_size)]
            dist.all_gather(memories, memory)
            record = {"step": step, "loss": float(totals[0] / totals[1]), "grad_norm": float(grad),
                      "epoch": state["epoch"], "batch_offset": state["batch_offset"],
                      "valid_action_coordinates": valid_count, "lr": optimizer.param_groups[0]["lr"],
                      "step_seconds": update_seconds, "global_batch_size": global_batch,
                      "peak_gpu_gib": [float(value[0]) for value in memories],
                      "peak_reserved_gpu_gib": [float(value[1]) for value in memories]}
            if validation is not None and step % int(config.get("eval_every", 100)) == 0:
                record["validation_bc_loss"] = validate_distributed(
                    model, validation, device, batch_size=batch_size, max_batches=int(config.get("eval_batches", 8)),
                    rank=rank, world_size=world_size)
            if step % int(config.get("save_every", 100)) == 0 or step == max_steps:
                checkpoint_dir = output / "checkpoints" / f"global_step_{step}" / "distributed"
                checkpoint_start = time.monotonic()
                save_distributed_checkpoint(model, optimizer, scheduler, checkpoint_dir, dict(state), metadata,
                                            rank=rank, world_size=world_size, device=device)
                record["checkpoint_seconds"] = time.monotonic() - checkpoint_start
                if rank == 0:
                    (output / "latest_checkpoint.txt").write_text(str(checkpoint_dir) + "\n")
            record["iteration_wall_seconds"] = time.monotonic() - start
            if rank == 0:
                print(json.dumps(record), flush=True)
                with (output / "metrics.jsonl").open("a") as log:
                    log.write(json.dumps(record) + "\n")
        if config.get("export", True):
            export_start = time.monotonic()
            exported = export_distributed_hf(
                model, policy, output / "checkpoints" / f"global_step_{state['global_step']}" / "hf_ckpt",
                dtype=getattr(torch, config.get("export_dtype", "bfloat16")), processor=processor, rank=rank)
            if rank == 0:
                print(json.dumps({"exported_hf_ckpt": str(exported), "export_seconds": time.monotonic() - export_start}), flush=True)
        dist.barrier()
        return state
    except RuntimeError as error:
        # Print before process-group cleanup: another rank may be blocked in a
        # collective when this rank exhausts CUDA memory. The parent launcher
        # can then stop the whole attempt and retry a smaller physical batch.
        if cuda_oom_error(error):
            print(json.dumps({"event": "LRVLA_CUDA_OOM", "rank": rank,
                              "completed_steps": locals().get("state", {}).get("global_step", 0),
                              "global_batch_size": global_batch,
                              "error": str(error)}), file=sys.stderr, flush=True)
        raise
    finally:
        dist.destroy_process_group()


def cuda_oom_error(error):
    message = str(error).lower()
    return any(pattern in message for pattern in (
        "cuda out of memory", "cuda error: out of memory", "cuda_error_out_of_memory",
        "cuda failure 'out of memory'",
    ))
