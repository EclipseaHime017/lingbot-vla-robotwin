"""Single-GPU, clean-only supervised post-training of LingBot-VLA v2."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import shutil
import time

import torch
from torch.utils.data import DataLoader
import yaml

from .training_checkpoint import export_hf_checkpoint, resume_training_checkpoint, save_training_checkpoint
from .training_data import build_clean_datasets, epoch_dataloader, model_collate
from .training_models import configure_trainable, flow_matching_bc, memory_report, normalize_bc_gradients
from .training_identity import base_checkpoint_identity, manifest_content_sha256
from .training_runtime import add_vendor_path, build_policy, enable_action_checkpointing


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(value):
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (PROJECT_ROOT / path).resolve()


def move_batch(batch, device):
    return {key: value.to(device, non_blocking=True) for key, value in batch.items()}


def schedule_multiplier(step, max_steps, warmup, minimum=0.1):
    if warmup and step < warmup:
        return max(1, step + 1) / warmup
    progress = min(1, max(0, (step - warmup) / max(1, max_steps - warmup)))
    return minimum + (1 - minimum) * 0.5 * (1 + math.cos(math.pi * progress))


def validate(model, loader, device, max_batches=8, loss_fn=flow_matching_bc):
    model.eval()
    numerator, valid_count = 0.0, 0.0
    gpu_ids = [torch.device(device).index or 0] if str(device).startswith("cuda") else []
    with torch.random.fork_rng(devices=gpu_ids), torch.no_grad():
        torch.manual_seed(12345)
        for idx, batch in enumerate(loader):
            if idx >= max_batches:
                break
            with torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16,
                                enabled=torch.device(device).type == "cuda"):
                loss = loss_fn(model, move_batch(batch, device))
            count = float(batch["joint_mask"].float().sum())
            numerator += float(loss) * count
            valid_count += count
    model.train()
    return numerator / valid_count if valid_count else None


def prepare_runtime_files(config, architecture, manifest_path, norm_path, output):
    manifest_digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    provenance_path = norm_path.with_suffix(".provenance.json")
    if not provenance_path.is_file():
        raise ValueError("Norm statistics need a clean-data provenance sidecar; run scripts/prepare_data.py norm.")
    provenance = json.loads(provenance_path.read_text())
    content_digest = manifest_content_sha256(json.loads(manifest_path.read_text()))
    matches_manifest = (provenance.get("manifest_sha256") == manifest_digest or
                        provenance.get("manifest_content_sha256") == content_digest)
    if provenance.get("clean_only") is not True or not matches_manifest:
        raise ValueError("Norm provenance does not match this clean-only manifest.")
    if provenance.get("norm_stats_sha256") != hashlib.sha256(norm_path.read_bytes()).hexdigest():
        raise ValueError("Norm statistics changed after clean-only provenance was created.")
    holdout = int(config.get("val_episodes", 5))
    if provenance.get("val_episode_ids") != list(range(50 - holdout, 50)):
        raise ValueError("Norm train/validation episode split differs from the training config.")
    output.mkdir(parents=True, exist_ok=True)
    assets = output / "assets"
    assets.mkdir(exist_ok=True)
    saved_norm = assets / "norm_stats.json"
    if norm_path.resolve() != saved_norm.resolve():
        shutil.copy2(norm_path, saved_norm)
        shutil.copy2(provenance_path, saved_norm.with_suffix(".provenance.json"))
    saved_manifest = assets / "clean_manifest.json"
    if manifest_path.resolve() != saved_manifest.resolve():
        shutil.copy2(manifest_path, saved_manifest)
    data = dict(architecture["data"])
    data.update(img_size=int(config.get("img_size", 256)), norm_stats_file=str(saved_norm),
                image_augment=bool(config.get("image_augment", False)), video_backend="pyav", use_future_image=False,
                prompt_type="global", data_name="multi")
    for key in ("joints", "norm_type"):
        data[key] = [repr(item) if isinstance(item, dict) else item for item in data[key]]
    robot_dir = output / "configs" / "robot_configs"
    robot_dir.mkdir(parents=True, exist_ok=True)
    vendor = PROJECT_ROOT / "vendor" / "lingbot-vla-v2"
    robot_config = yaml.safe_load((vendor / "configs" / "robot_configs" / "robotwin.yaml").read_text())
    robot_config["norm_stats"] = "assets/norm_stats.json"
    robot_path = robot_dir / "robotwin.yaml"
    robot_path.write_text(yaml.safe_dump(robot_config, sort_keys=False))
    data["robot_config_root"] = str(robot_dir)
    data["train_path"] = str(saved_manifest)
    return data, robot_path, manifest_digest


def portable_inference_cli(cli):
    """Normalize deployment paths against the run directory, independent of cwd."""
    from copy import deepcopy
    result = deepcopy(cli)
    result["model"]["tokenizer_path"] = "assets/qwen3_vl"
    result["data"].update(norm_stats_file="assets/norm_stats.json", robot_config_root="configs/robot_configs",
                           train_path="assets/clean_manifest.json")
    return result


def train(config, *, dry_run=False):
    action_chunk_tail = config.get("action_chunk_tail", "mask")
    if action_chunk_tail not in {"mask", "drop"}:
        raise ValueError("action_chunk_tail must be 'mask' or 'drop'.")
    for key, default in (("batch_size", 1), ("gradient_accumulation", 16), ("max_steps", 1000),
                         ("save_every", 100), ("eval_every", 100)):
        if int(config.get(key, default)) <= 0:
            raise ValueError(f"{key} must be positive.")
    if not 0 <= int(config.get("val_episodes", 5)) < 50:
        raise ValueError("val_episodes must be between 0 and 49.")
    torch.manual_seed(int(config.get("seed", 42)))
    random.seed(int(config.get("seed", 42)))
    import numpy as np
    np.random.seed(int(config.get("seed", 42)))
    device = config.get("device", "cuda")
    if torch.device(device).type != "cuda" and not dry_run:
        raise ValueError("The full 6B model training entrypoint needs CUDA; use --smoke for CPU validation.")
    if torch.device(device).type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable in this environment.")
    base, qwen = resolve_path(config["base_checkpoint"]), resolve_path(config["qwen_path"])
    if "robotwin" in base.name.lower():
        raise ValueError("Use robbyant/lingbot-vla-v2-6b base weights; official RoboTwin post-trained initialization is forbidden.")
    if not (qwen / "config.json").is_file():
        raise FileNotFoundError(f"Qwen3-VL processor/config assets missing: {qwen}")
    architecture = yaml.safe_load(resolve_path(config["architecture_yaml"]).read_text())
    architecture["train"]["chunk_size"] = int(config.get("chunk_size", 50))
    if not dry_run:
        manifest_path, norm_path = resolve_path(config["manifest"]), resolve_path(config["norm_stats"])
        from .data import load_manifest
        manifest = load_manifest(manifest_path)
        if not manifest.get("complete_50_tasks") and not config.get("allow_subset", False):
            raise ValueError("Competition co-training requires all 50 clean tasks. Use --allow-subset only to debug the pipeline.")
        output = resolve_path(config["output_dir"])
        data_config, robot_config, manifest_digest = prepare_runtime_files(config, architecture, manifest_path, norm_path, output)
        print("Verifying frozen base shard SHA-256 identity before model loading...", flush=True)
        base_identity = base_checkpoint_identity(base)
    policy = build_policy(PROJECT_ROOT, architecture, base, qwen, device=device, load_weights=not dry_run)
    selection = configure_trainable(policy, config)
    report = memory_report(policy)
    concise_selection = {k: v for k, v in selection.items() if k not in {"trainable_names", "lora_modules"}}
    concise_selection["lora_module_count"] = len(selection["lora_modules"])
    print(json.dumps({"selection": concise_selection,
                      "memory": report, "dry_run": dry_run}, ensure_ascii=False), flush=True)
    if dry_run:
        return report
    total_gpu_gib = torch.cuda.get_device_properties(torch.device(device)).total_memory / 2**30
    if report["adamw_training_lower_bound_gib"] > total_gpu_gib * 0.92:
        raise RuntimeError(f"Weight/gradient/AdamW lower bound {report['adamw_training_lower_bound_gib']:.2f} GiB exceeds safe space on {total_gpu_gib:.2f} GiB GPU. Use LoRA or fewer expert_last_n_layers, or move to a larger server GPU.")
    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(qwen, padding_side="right", local_files_only=True)
    datasets, validation, splits = build_clean_datasets(manifest, data_config, policy.config, processor,
                                                       robot_config, int(config.get("val_episodes", 5)),
                                                       action_chunk_tail=action_chunk_tail)
    print(json.dumps({"training_samples": len(datasets),
                      "validation_samples": len(validation) if validation is not None else 0,
                      "action_chunk_tail": action_chunk_tail}, ensure_ascii=False), flush=True)
    if config.get("gradient_checkpointing", True):
        enable_action_checkpointing(policy)
    train_values = {**architecture["train"], **policy.config.to_dict()}
    train_values.update(output_dir=str(output), train_expert_only=True, freeze_vision_encoder=True,
                        attention_implementation="eager", vit_attn_implementation="sdpa", use_compile=False)
    cli = {"model": {**architecture["model"], "model_path": str(base), "tokenizer_path": str(qwen)},
           "train": train_values, "data": data_config}
    # Qwen processor/config assets are small and contain no duplicate VLM weights.
    qwen_assets = output / "assets" / "qwen3_vl"
    processor.save_pretrained(qwen_assets)
    shutil.copy2(qwen / "config.json", qwen_assets / "config.json")
    saved_cli = portable_inference_cli(cli)
    saved_cli["train"]["tokenizer_path"] = "assets/qwen3_vl"
    (output / "lingbotvla_cli.yaml").write_text(yaml.safe_dump(saved_cli, sort_keys=False))
    (output / "training_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    (output / "episode_splits.json").write_text(json.dumps(splits, indent=2))
    metadata = {"base_checkpoint": str(base), "base_identity": base_identity,
                "manifest_sha256": manifest_digest, "manifest_content_sha256": manifest_content_sha256(manifest),
                "selection": selection,
                "training_method": "clean_only_behavior_cloning", "competition_all_50_tasks": bool(manifest.get("complete_50_tasks")),
                "loss_normalization": "valid_action_coordinate_mean_v1",
                "data_sampling": {"action_chunk_tail": action_chunk_tail, "validation_action_chunk_tail": "mask",
                                  "val_episodes": int(config.get("val_episodes", 5)),
                                  "seed": int(config.get("seed", 42)), "batch_size": int(config.get("batch_size", 1)),
                                  "gradient_accumulation": int(config.get("gradient_accumulation", 16))},
                "memory": report, "norm_sha256": hashlib.sha256(norm_path.read_bytes()).hexdigest(),
                "qwen_config_sha256": hashlib.sha256((qwen / "config.json").read_bytes()).hexdigest(),
                "model_geometry": {"chunk_size": policy.config.chunk_size, "max_action_dim": policy.config.max_action_dim,
                                   "max_state_dim": policy.config.max_state_dim, "img_size": data_config["img_size"]}}
    (output / "run_metadata.json").write_text(json.dumps(metadata, indent=2))
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=float(config.get("lr", 1e-4)),
                                  weight_decay=float(config.get("weight_decay", 0.01)), foreach=False)
    max_steps = int(config.get("max_steps", 1000))
    warmup = min(int(config.get("warmup_steps", 100)), max_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda step: schedule_multiplier(step, max_steps, warmup))
    state = {"global_step": 0, "epoch": 0, "batch_offset": 0}
    if config.get("resume"):
        state = resume_training_checkpoint(policy, optimizer, scheduler, resolve_path(config["resume"]), expected_metadata=metadata)
    batch_size, accumulation = int(config.get("batch_size", 1)), int(config.get("gradient_accumulation", 16))
    if max_steps <= 0 or batch_size <= 0 or accumulation <= 0:
        raise ValueError("max_steps, batch_size and gradient_accumulation must be positive.")
    workers = int(config.get("num_workers", 0))
    if workers > 0 and config.get("resume") and config.get("image_augment", False):
        print("Resume preserves consumed batches and main RNG; worker augmentation RNG replay is not guaranteed.", flush=True)

    def iterator():
        return iter(epoch_dataloader(datasets, batch_size, int(config.get("seed", 42)),
                                    state["epoch"], state["batch_offset"], workers))

    batches = iterator()
    validation_loader = DataLoader(validation, batch_size=batch_size, collate_fn=model_collate, num_workers=0) if validation else None
    policy.train()
    optimizer.zero_grad(set_to_none=True)
    metrics_path = output / "metrics.jsonl"
    while state["global_step"] < max_steps:
        start = time.monotonic()
        loss_numerator, valid_count = 0.0, 0.0
        for _ in range(accumulation):
            try:
                batch = next(batches)
            except StopIteration:
                state["epoch"] += 1
                state["batch_offset"] = 0
                batches = iterator()
                batch = next(batches)
            state["batch_offset"] += 1
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = flow_matching_bc(policy, move_batch(batch, device), reduction="sum")
            if not torch.isfinite(loss):
                raise FloatingPointError(f"Nonfinite BC loss at step {state['global_step']}")
            loss.backward()
            loss_numerator += float(loss.detach())
            valid_count += float(batch["joint_mask"].float().sum())
        normalize_bc_gradients(params, valid_count)
        grad = torch.nn.utils.clip_grad_norm_(params, float(config.get("max_grad_norm", 1)))
        if not torch.isfinite(grad):
            raise FloatingPointError("Nonfinite gradient norm; optimizer step aborted.")
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        state["global_step"] += 1
        step = state["global_step"]
        record = {"step": step, "loss": loss_numerator / valid_count, "grad_norm": float(grad),
                  "valid_action_coordinates": valid_count,
                  "lr": optimizer.param_groups[0]["lr"], "step_seconds": time.monotonic() - start,
                  "peak_gpu_gib": torch.cuda.max_memory_allocated(torch.device(device)) / 2**30}
        if validation_loader and step % int(config.get("eval_every", 100)) == 0:
            record["validation_bc_loss"] = validate(policy, validation_loader, device, int(config.get("eval_batches", 8)))
        print(json.dumps(record), flush=True)
        with metrics_path.open("a") as log:
            log.write(json.dumps(record) + "\n")
        if step % int(config.get("save_every", 100)) == 0 or step == max_steps:
            checkpoint_dir = output / "checkpoints" / f"global_step_{step}"
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
            save_training_checkpoint(policy, optimizer, scheduler, checkpoint_dir / "training.pt", dict(state), metadata)
            (output / "latest_checkpoint.txt").write_text(str(checkpoint_dir / "training.pt") + "\n")
    if config.get("export", True):
        checkpoint_dir = output / "checkpoints" / f"global_step_{state['global_step']}"
        exported = export_hf_checkpoint(policy, checkpoint_dir / "hf_ckpt",
                                        dtype=getattr(torch, config.get("export_dtype", "bfloat16")), processor=processor)
        print(json.dumps({"exported_hf_ckpt": str(exported), "official_cli": str(output / "lingbotvla_cli.yaml")}), flush=True)
    return state


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_lora.yaml")
    parser.add_argument("--device", choices=["cpu", "cuda"])
    parser.add_argument("--base-checkpoint")
    parser.add_argument("--qwen-path")
    parser.add_argument("--manifest")
    parser.add_argument("--norm-stats")
    parser.add_argument("--output-dir")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--gradient-accumulation", type=int)
    parser.add_argument("--expert-last-n-layers", type=int)
    parser.add_argument("--val-episodes", type=int)
    parser.add_argument("--resume")
    parser.add_argument("--dry-run", action="store_true", help="Construct on meta and report parameter memory; no 6B weights loaded.")
    parser.add_argument("--allow-subset", action="store_true", help="Debug with fewer than 50 tasks; resulting run is marked as a subset.")
    parser.add_argument("--smoke", action="store_true", help="Train/export/reload a tiny synthetic model, without using competition assets.")
    args = parser.parse_args(argv)
    if args.smoke:
        from .training_smoke import run_smoke
        return run_smoke(resolve_path(args.output_dir or "artifacts/training_smoke"), args.device or "cpu", args.max_steps or 30)
    config = yaml.safe_load(resolve_path(args.config).read_text())
    for key, value in vars(args).items():
        if key not in {"config", "dry_run", "smoke", "allow_subset"} and value is not None:
            config[key] = value
    if args.allow_subset:
        config["allow_subset"] = True
    return train(config, dry_run=args.dry_run)
