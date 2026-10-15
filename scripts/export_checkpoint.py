#!/usr/bin/env python3
"""Export saved FP32 masters over the original base without loading Adam/GPU."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.training_checkpoint import export_hf_checkpoint, overlay_base_state_items  # noqa: E402
from lrvla.training_distributed_checkpoint import load_distributed_export  # noqa: E402


def export_saved(checkpoint, base, output=None, dtype=torch.float32):
    checkpoint, base = Path(checkpoint).resolve(), Path(base).resolve()
    run = checkpoint.parent.parent.parent
    output = Path(output).resolve() if output else checkpoint.parent / f"hf_ckpt_{str(dtype).split('.')[-1]}"
    if output.parent != checkpoint.parent:
        raise ValueError("Keep exports beside training.pt/distributed so run assets remain portable.")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite an existing export: {output}")
    cli = yaml.safe_load((run / "lingbotvla_cli.yaml").read_text())
    assets = run / "assets/qwen3_vl"
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json", "preprocessor_config.json"):
        if not (assets / name).is_file():
            raise FileNotFoundError(f"Missing portable processor resource: {assets / name}")
    if checkpoint.is_dir():
        masters, payload = load_distributed_export(checkpoint)
    else:
        payload = torch.load(checkpoint, map_location="cpu", weights_only=True, mmap=True)
        masters = payload["trainable_state"]
    metadata = payload["metadata"]
    values = {**cli["model"], **cli["train"]}
    for key in ("chunk_size", "max_action_dim", "max_state_dim"):
        if values.get(key) != metadata["model_geometry"][key]:
            raise ValueError(f"Export CLI differs from checkpoint model_geometry: {key}")
    if cli["data"]["img_size"] != metadata["model_geometry"]["img_size"]:
        raise ValueError("Export CLI image size differs from checkpoint.")
    if hashlib.sha256((assets / "config.json").read_bytes()).hexdigest() != metadata["qwen_config_sha256"]:
        raise ValueError("Export processor config differs from checkpoint qwen_config_sha256.")
    norm = run / "assets/norm_stats.json"
    if not norm.is_file() or hashlib.sha256(norm.read_bytes()).hexdigest() != metadata["norm_sha256"]:
        raise ValueError("Export normalization differs from checkpoint norm_sha256.")
    items = overlay_base_state_items(base, masters, metadata["selection"],
                                    expected_base_identity=metadata["base_identity"])
    values["torch_dtype"] = str(dtype).split(".")[-1]
    config = SimpleNamespace(model_type=values.get("model_type", "lingbot_vla_v2"),
                             to_json_file=lambda path: Path(path).write_text(json.dumps(values, indent=2)))
    model = torch.nn.Module()
    model.config = config
    exported = export_hf_checkpoint(model, output, dtype=dtype, state_items=items)
    for asset in assets.iterdir():
        if asset.is_file():
            shutil.copy2(asset, exported / asset.name)
    # The VLA config, rather than Qwen's processor-side config, belongs here.
    config.to_json_file(exported / "config.json")
    record = json.loads((exported / "export.json").read_text())
    record.update(frozen_source="original_base", trainable_source="fp32_master",
                  base_identity=metadata["base_identity"], source_checkpoint=str(checkpoint),
                  source_step=payload["state"]["global_step"],
                  merged_lora=bool(metadata["selection"].get("lora_modules")),
                  lora_modules=len(metadata["selection"].get("lora_modules", [])))
    (exported / "export.json").write_text(json.dumps(record, indent=2) + "\n")
    return exported


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()
    print(export_saved(args.checkpoint, args.base_checkpoint, args.output, getattr(torch, args.dtype)))


if __name__ == "__main__":
    main()
