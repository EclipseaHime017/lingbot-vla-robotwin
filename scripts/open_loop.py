#!/usr/bin/env python3
"""Evaluate action predictions on held-out clean episodes with the official V2 server."""
import argparse
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import load_manifest
from lrvla.wsl_tools import linux_executable


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/clean_manifest.json")
    parser.add_argument("--task", default="adjust_bottle")
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--max-infer-time", type=int, default=10, help="Maximum action-chunk forwards per held-out episode")
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/open_loop")
    parser.add_argument("--qwen", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--native-server", action="store_true", help="Use upstream full model constructor")
    args = parser.parse_args()
    if not 1 <= args.val_episodes < 50:
        parser.error("--val-episodes must be 1..49")
    if args.max_infer_time < 1:
        parser.error("--max-infer-time must be positive")
    manifest = load_manifest(args.manifest)
    entry = next((entry for entry in manifest["datasets"] if entry["task"] == args.task), None)
    if entry is None:
        parser.error("Task is absent from clean manifest")
    vendor = ROOT / "vendor/lingbot-vla-v2"
    checkpoint = args.checkpoint.resolve()
    model_run = checkpoint.parent.parent.parent
    eval_script = vendor / "scripts/open_loop_eval.py"
    if not args.native_server:
        # Adapt only model loading; preserve official action metric/plot code.
        runtime = args.output.resolve() / "runtime"
        runtime.mkdir(parents=True, exist_ok=True)
        source = eval_script.read_text()
        start = source.index("def load_policy_server(")
        stop = source.index("\ndef str2bool(", start)
        source = source[:start] + "def load_policy_server(policy, model_path):\n    from lrvla.training_inference import LingbotVLAv2Server\n    return LingbotVLAv2Server\n\n" + source[stop:]
        eval_script = runtime / "open_loop_eval.py"
        eval_script.write_text(source)
    python = linux_executable("python", preferred=Path(sys.executable))
    command = [str(python), str(eval_script), "--model_path", str(checkpoint), "--robo_name", "robotwin",
               "--data_path", entry["path"], "--policy", "qwen3vl", "--use_length", "50",
               "--max_infer_time", str(args.max_infer_time),
               "--traj_ids", *map(str, range(50 - args.val_episodes, 50)), "--save_plot_path", str(args.output.resolve())]
    if args.precision == "bf16":
        command.append("--use_bf16")
    env = dict(os.environ, QWEN3VL_PATH=str(args.qwen.resolve()), PYTHONNOUSERSITE="1", PYTHONPATH=os.pathsep.join([str(ROOT / "src"), str(vendor)]))
    env["PATH"] = os.pathsep.join(filter(None, [str(python.parent), env.get("PATH", "")]))
    if args.dry_run:
        print(command)
        return
    print("Open-loop held-out action plots are diagnostics; simulator success must be evaluated separately.")
    subprocess.run(command, cwd=model_run, env=env, check=True)


if __name__ == "__main__":
    main()
