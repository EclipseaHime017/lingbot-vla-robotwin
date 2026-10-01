#!/usr/bin/env python3
"""Run one fixed exported V2 policy on official clean/randomized RoboTwin scenes.

FP32, both settings, 50 tasks x 100 trials is the release comparison protocol.
BF16/subsets/short trials are explicitly recorded as approximate smoke checks.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import parse_tasks
from lrvla.evaluation import INSTRUCTIONS, ROBOTWIN_REVISION, aggregate_results, create_eval_runtime, export_official_results
from lrvla.training_identity import base_checkpoint_identity
from lrvla.vendor_sources import upstream_revision
from lrvla.wsl_tools import linux_executable


def default_conda() -> Path:
    return Path(os.environ.get("CONDA_ROOT") or Path.home() / "miniconda3").expanduser() / "bin/conda"


def stop(process):
    if process and process.poll() is None:
        os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="Exported run/checkpoints/global_step_N/hf_ckpt")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--setting", choices=["both", *INSTRUCTIONS], default="both")
    parser.add_argument("--trials", type=int, default=100)
    parser.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--smoke", action="store_true", help="One task, two trials, BF16; explicitly approximate")
    parser.add_argument("--robotwin", type=Path, default=ROOT / "vendor/RoboTwin")
    parser.add_argument("--qwen", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    parser.add_argument("--conda", type=Path, default=default_conda(), help="Conda installation path; locates Linux environment interpreters")
    parser.add_argument("--inference-env", default="lingbot-vla")
    parser.add_argument("--sim-env", default="robotwin-sim")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--port", type=int, default=9330)
    parser.add_argument("--use-length", type=int, default=50)
    parser.add_argument("--compile", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--native-server", action="store_true", help="Use upstream full-FP32 constructor; needs substantial host RAM")
    parser.add_argument("--server-timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--team-id", default="replace_with_your_team_id")
    args = parser.parse_args()
    if args.smoke:
        args.tasks, args.trials, args.precision = "adjust_bottle", 2, "bf16"
    tasks = parse_tasks(args.tasks)
    settings = list(INSTRUCTIONS) if args.setting == "both" else [args.setting]
    if not 1 <= args.trials <= 100:
        parser.error("--trials must be 1..100")
    if args.precision == "bf16":
        print("BF16 is an approximate pipeline check; official published validation uses FP32 and roughly 32 GB with simulation.")
    backend = "native" if args.native_server else "streamed_sdpa_reference_moe"
    if not args.native_server:
        print("The streamed server uses SDPA/reference MoE kernels; equivalence to the official FP32 server remains unverified.")
    run = args.output or ROOT / "runs/evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        report = aggregate_results(run, tasks, settings, args.precision, args.trials, backend)
        export_official_results(report, run / "official_results.json", args.team_id, ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json")
        print(json.dumps(report, indent=2))
        return
    checkpoint = args.checkpoint.resolve()
    model_run = checkpoint.parent.parent.parent
    vendor = ROOT / "vendor/lingbot-vla-v2"
    robotwin = args.robotwin.resolve()
    conda_path = args.conda.expanduser().resolve()
    conda_root = conda_path.parent.parent
    infer_python = linux_executable("python", preferred=conda_root / "envs" / args.inference_env / "bin/python")
    sim_python = linux_executable("python", preferred=conda_root / "envs" / args.sim_env / "bin/python")
    client = create_eval_runtime(vendor, robotwin, run / "runtime", args.trials)
    shared_environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, QWEN3VL_PATH=str(args.qwen.resolve()), PYTHONNOUSERSITE="1",
                              PYTHONUNBUFFERED="1", SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0", TOKENIZERS_PARALLELISM="false")
    shared_environment["PYTHONPATH"] = os.pathsep.join([str(run / "runtime"), str(vendor), str(robotwin), str(robotwin / "description/utils")])
    env = dict(shared_environment)
    env["PATH"] = os.pathsep.join(filter(None, [str(infer_python.parent), shared_environment.get("PATH", "")]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(conda_root / "envs" / args.inference_env / "lib"), shared_environment.get("LD_LIBRARY_PATH", "")]))
    sim_environment = dict(shared_environment)
    sim_environment["PATH"] = os.pathsep.join(filter(None, [str(sim_python.parent), shared_environment.get("PATH", "")]))
    sim_environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(conda_root / "envs" / args.sim_env / "lib"), shared_environment.get("LD_LIBRARY_PATH", "")]))
    server_entry = ["-m", "deploy.lingbot_vla_v2_policy"] if args.native_server else [str(ROOT / "scripts/serve_streamed.py")]
    infer = [str(infer_python), *server_entry,
             "--model_path", str(checkpoint), "--use_length", str(args.use_length), "--use_bf16", str(args.precision == "bf16"),
             "--use_fp32", str(args.precision == "fp32"), "--use_compile", str(args.compile), "--port", str(args.port)]
    commands = []
    for setting in settings:
        for task in tasks:
            command = [str(sim_python), "-u", str(client),
                       "--config", "policy/pi0/deploy_policy.yml", "--overrides", "--task_name", task, "--task_config", setting,
                       "--ckpt_setting", "cotrain", "--train_config_name", "0", "--seed", "0", "--policy_name", "pi0",
                       "--port", str(args.port), "--robo_name", "robotwin", "--instruction_type", INSTRUCTIONS[setting],
                       "--eval_video_log", str(args.video), "--output_dir", str(run / setting / "eval_results")]
            commands.append({"setting": setting, "task": task, "command": command})
    plan = {"checkpoint": str(checkpoint), "model_run": str(model_run), "precision": args.precision, "trials": args.trials,
            "tasks": tasks, "settings": settings, "inference_command": infer, "evaluation_commands": commands,
            "instruction_types": INSTRUCTIONS, "expert_starts_per_scene": 5, "fixed_checkpoint": True,
            "full_protocol": len(tasks) == 50 and args.trials == 100 and len(settings) == 2 and args.precision == "fp32",
            "schema": "official schema_version=1; precision and protocol recorded in sidecar"}
    (run / "evaluation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        print(f"Dry-run plan: {run / 'evaluation_plan.json'}")
        return
    actual_rev = upstream_revision(robotwin)
    if actual_rev != ROBOTWIN_REVISION:
        raise RuntimeError(f"RoboTwin revision mismatch: {actual_rev}")
    if not (model_run / "lingbotvla_cli.yaml").is_file() or not list(checkpoint.glob("*.safetensors")):
        raise RuntimeError("Checkpoint must be a full exported hf_ckpt with run/lingbotvla_cli.yaml")
    if not (model_run / "configs/robot_configs/robotwin.yaml").is_file():
        raise RuntimeError("Export must carry run/configs/robot_configs/robotwin.yaml with clean normalization")
    # Isolate native Vulkan rendering before loading even a single model tensor.
    # WSL CUDA works independently from the renderer required by RoboTwin.
    from doctor import isolated_probe, RENDER_CODE
    preflight = isolated_probe(sim_python, RENDER_CODE, timeout=30, cwd=robotwin, env=sim_environment)
    (run / "render_preflight.json").write_text(json.dumps(preflight, indent=2) + "\n")
    if not preflight["ok"]:
        raise RuntimeError(f"RoboTwin rendering preflight {preflight['status']}; read {run / 'render_preflight.json'}. Model loading was not started.")
    with socket.socket() as port_check:
        try:
            port_check.bind(("127.0.0.1", args.port))
        except OSError as error:
            raise RuntimeError(f"Inference port {args.port} is already in use") from error
    plan["checkpoint_identity"] = base_checkpoint_identity(checkpoint)
    (run / "evaluation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    server = worker = None
    try:
        with (run / "inference.log").open("w") as log:
            server = subprocess.Popen(infer, cwd=model_run, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            deadline = time.monotonic() + args.server_timeout
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Inference server exited; read {run / 'inference.log'}")
                try:
                    with socket.create_connection(("localhost", args.port), timeout=1):
                        break
                except OSError:
                    time.sleep(2)
            else:
                raise TimeoutError(f"Inference server did not become ready; read {run / 'inference.log'}")
            for item in commands:
                logfile = run / item["setting"] / "eval_logs" / f"{item['task']}.log"
                logfile.parent.mkdir(parents=True, exist_ok=True)
                with logfile.open("w") as output:
                    worker = subprocess.Popen(item["command"], cwd=robotwin, env=sim_environment, stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    code = worker.wait()
                logfile.with_suffix(".status.json").write_text(json.dumps({"exit_code": code}) + "\n")
                print(f"{item['setting']}/{item['task']}: exit={code}", flush=True)
    finally:
        stop(worker)
        stop(server)
        report = aggregate_results(run, tasks, settings, args.precision, args.trials, backend)
        export_official_results(report, run / "official_results.json", args.team_id, ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json")
    report = json.loads((run / "results.json").read_text())
    print(f"Results: {run / 'results.json'} ({report['status']})")
    if not report["complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
