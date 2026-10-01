#!/usr/bin/env python3
"""Measure real clean-data CUDA BC updates without writing training checkpoints.

Each invocation profiles one configuration in a fresh process. CPU video decode
and tokenization happen before timing; timed updates include host-to-device copy,
forward, backward, gradient clipping, AdamW and zero_grad. The result therefore
describes model throughput, rather than end-to-end data-loading throughput.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import random
import statistics
import sys
import time
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def resolve(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (ROOT / path).resolve()


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--mode", choices=("lora", "expert"), default="lora")
    result.add_argument("--batch-size", type=int, default=1)
    result.add_argument("--expert-last-n-layers", type=int, default=2)
    result.add_argument("--lora-rank", type=int, default=8)
    result.add_argument("--steps", type=int, default=3)
    result.add_argument("--warmup", type=int, default=1)
    result.add_argument("--gradient-accumulation", type=int, default=1)
    result.add_argument("--task", default="adjust_bottle")
    result.add_argument("--manifest", default="data/clean_manifest.json")
    result.add_argument("--norm-stats", default="data/clean_norm_stats.json")
    result.add_argument("--base-checkpoint", default="models/lingbot-vla-v2-6b")
    result.add_argument("--qwen-path", default="models/Qwen3-VL-4B-Instruct")
    result.add_argument("--architecture-yaml", default="vendor/lingbot-vla-v2/configs/vla/robotwin/robotwin.yaml")
    result.add_argument("--img-size", type=int, default=256)
    result.add_argument("--chunk-size", type=int, default=50)
    result.add_argument("--seed", type=int, default=42)
    result.add_argument("--allocator-budget-gib", type=float,
                        help="Optional PyTorch allocator cap on this GPU; this is not a benchmark on another GPU.")
    result.add_argument("--output", required=True)
    return result


def profile(args, report):
    import numpy as np
    import torch
    import yaml
    from transformers import AutoProcessor
    from lrvla.data import load_manifest
    from lrvla.training_data import build_clean_datasets, model_collate
    from lrvla.training_models import configure_trainable, flow_matching_bc, memory_report
    from lrvla.training_runtime import build_policy, enable_action_checkpointing

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; run with the WSL GPU accessible.")
    for name in ("batch_size", "steps", "gradient_accumulation", "img_size", "chunk_size", "lora_rank"):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.warmup < 1:
        raise ValueError("At least one warmup update is needed to initialize AdamW state.")
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    report["runtime"] = {"gpu": properties.name, "gpu_total_gib": properties.total_memory / 2**30,
                         "torch": torch.__version__, "cuda": torch.version.cuda,
                         "python": sys.version, "weight_dtype": "bfloat16", "trainable_dtype": "float32",
                         "optimizer": "AdamW FP32, foreach=False", "gradient_checkpointing": True,
                         "attention": "PyTorch SDPA", "moe": "Torch reference"}
    if args.allocator_budget_gib is not None:
        if not 0 < args.allocator_budget_gib <= properties.total_memory / 2**30:
            raise ValueError("Allocator budget must be positive and at most this GPU's total memory.")
        torch.cuda.set_per_process_memory_fraction(args.allocator_budget_gib * 2**30 / properties.total_memory, device)

    architecture = yaml.safe_load(resolve(args.architecture_yaml).read_text())
    architecture["train"]["chunk_size"] = args.chunk_size
    report["stage"] = "load_base"
    print(json.dumps({"stage": report["stage"]}), flush=True)
    load_start = time.perf_counter()
    policy = build_policy(ROOT, architecture, resolve(args.base_checkpoint), resolve(args.qwen_path), device=str(device))
    config = {"mode": args.mode, "expert_last_n_layers": args.expert_last_n_layers,
              "lora_rank": args.lora_rank, "lora_alpha": args.lora_rank * 2,
              "lora_dropout": 0, "lora_targets": ["q_proj", "k_proj", "v_proj", "o_proj"],
              "train_projections": True}
    selection = configure_trainable(policy, config)
    report["selection"] = {key: value for key, value in selection.items() if key not in {"trainable_names", "lora_modules"}}
    report["selection"]["lora_module_count"] = len(selection["lora_modules"])
    report["memory_lower_bound"] = memory_report(policy)
    torch.cuda.synchronize(device)
    report["load_seconds"] = time.perf_counter() - load_start
    enable_action_checkpointing(policy)
    params = [parameter for parameter in policy.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=1e-4 if args.mode == "lora" else 5e-5,
                                  weight_decay=0.01, foreach=False)

    data_config = dict(architecture["data"])
    data_config.update(img_size=args.img_size, norm_stats_file=str(resolve(args.norm_stats)),
                       image_augment=False, video_backend="pyav", use_future_image=False,
                       prompt_type="global", data_name="multi")
    for name in ("joints", "norm_type"):
        data_config[name] = [repr(item) if isinstance(item, dict) else item for item in data_config[name]]
    manifest = load_manifest(resolve(args.manifest))
    manifest = {**manifest, "datasets": [entry for entry in manifest["datasets"] if entry["task"] == args.task]}
    if not manifest["datasets"]:
        raise ValueError(f"Task {args.task!r} is absent from this converted clean manifest.")
    processor = AutoProcessor.from_pretrained(resolve(args.qwen_path), padding_side="right", local_files_only=True)
    robot_config = ROOT / "vendor/lingbot-vla-v2/configs/robot_configs/robotwin.yaml"
    datasets, _, splits = build_clean_datasets(manifest, data_config, policy.config, processor, robot_config, 5)
    count = (args.warmup + args.steps) * args.gradient_accumulation * args.batch_size
    if count > len(datasets):
        raise ValueError(f"Need {count} distinct frames, but the training split has {len(datasets)} frames.")
    indices = random.Random(args.seed).sample(range(len(datasets)), count)
    report["stage"] = "prepare_cpu_batches"
    print(json.dumps({"stage": report["stage"], "distinct_frames": count}), flush=True)
    prepare_start = time.perf_counter()
    cpu_batches = []
    for offset in range(0, count, args.batch_size):
        cpu_batches.append(model_collate([datasets[index] for index in indices[offset:offset + args.batch_size]]))
    report["data"] = {"task": args.task, "frames_in_training_split": len(datasets), "distinct_profile_frames": count,
                      "episode_splits": splits, "frame_indices": indices, "cpu_prepare_seconds": time.perf_counter() - prepare_start,
                      "longest_padded_language_tokens": max(batch["lang_tokens"].shape[1] for batch in cpu_batches),
                      "longest_valid_language_tokens": max(int(batch["lang_masks"].sum(dim=-1).max()) for batch in cpu_batches),
                      "images_shape": list(cpu_batches[0]["images"].shape), "actions_shape": list(cpu_batches[0]["actions"].shape),
                      "scope": "One task, distinct real frames; not a peak-memory measurement over all 50 task prompts."}
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    records = []
    batch_iter = iter(cpu_batches)
    report["updates"] = records
    for step in range(args.warmup + args.steps):
        report["stage"] = "warmup" if step < args.warmup else "measure"
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
        start = time.perf_counter()
        losses = []
        for _ in range(args.gradient_accumulation):
            batch = {key: value.to(device, non_blocking=False) for key, value in next(batch_iter).items()}
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = flow_matching_bc(policy, batch)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite BC loss.")
            (loss / args.gradient_accumulation).backward()
            losses.append(float(loss.detach()))
            del batch, loss
        grad = torch.nn.utils.clip_grad_norm_(params, 1.0)
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradient norm.")
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize(device)
        seconds = time.perf_counter() - start
        record = {"step": step + 1, "warmup": step < args.warmup, "optimizer_update_seconds": seconds,
                  "seconds_per_microbatch": seconds / args.gradient_accumulation,
                  "samples_per_second": args.batch_size * args.gradient_accumulation / seconds,
                  "loss": statistics.mean(losses), "grad_norm": float(grad),
                  "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 2**30,
                  "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 2**30,
                  "after_update_allocated_gib": torch.cuda.memory_allocated(device) / 2**30,
                  "after_update_reserved_gib": torch.cuda.memory_reserved(device) / 2**30}
        records.append(record)
        print(json.dumps(record), flush=True)
    measured = [item for item in records if not item["warmup"]]
    report["summary"] = {"optimizer_update_seconds_median": statistics.median(item["optimizer_update_seconds"] for item in measured),
                         "optimizer_update_seconds_mean": statistics.mean(item["optimizer_update_seconds"] for item in measured),
                         "samples_per_second_mean": statistics.mean(item["samples_per_second"] for item in measured),
                         "peak_allocated_gib": max(item["peak_allocated_gib"] for item in records),
                         "peak_reserved_gib": max(item["peak_reserved_gib"] for item in records),
                         "timing_scope": "Synchronized model updates plus CPU-to-GPU copies; excludes CPU decode, loading and exports."}
    report["status"] = "success"
    report["stage"] = "complete"


def main(argv=None):
    args = parser().parse_args(argv)
    output = resolve(args.output)
    report = {"schema_version": 1, "timestamp_utc": datetime.now(timezone.utc).isoformat(),
              "status": "started", "configuration": vars(args),
              "effective_batch_size": args.batch_size * args.gradient_accumulation,
              "writes_training_checkpoints": False}
    exit_code = 0
    try:
        profile(args, report)
    except Exception as exc:
        import torch
        oom = isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()
        report["status"] = "oom" if oom else "error"
        report["error_type"], report["error"] = type(exc).__name__, str(exc)
        report["traceback"] = traceback.format_exc()
        if torch.cuda.is_available():
            report["failure_memory"] = {"peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
                                        "peak_reserved_gib": torch.cuda.max_memory_reserved() / 2**30}
        exit_code = 2 if oom else 1
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2))
    temporary.replace(output)
    print(json.dumps({"status": report["status"], "output": str(output)}), flush=True)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
