#!/usr/bin/env python3
"""Run one fixed exported V2 policy on official clean/randomized RoboTwin scenes.

FP32, both settings, 50 tasks x 100 trials is the release comparison protocol.
BF16/subsets/short trials are explicitly recorded as approximate smoke checks.
"""
# ruff: noqa: E402
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time
import yaml

ROOT = Path(__file__).resolve().parents[1]
RENDER_BINDING = True
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import parse_tasks
from lrvla.evaluation import (
    EXPERT_SEED_POLICY, INSTRUCTIONS, ROBOTWIN_REVISION, aggregate_results, create_eval_runtime,
    evaluation_task_complete, export_official_results,
)
from lrvla.training_identity import base_checkpoint_identity
from lrvla.simulation_workers import run_clients
from lrvla.policy_sessions import POLICY_RNG
from lrvla.vendor_sources import upstream_revision
from lrvla.wsl_tools import linux_executable


def default_conda() -> Path:
    return Path(os.environ.get("CONDA_ROOT") or Path.home() / "miniconda3").expanduser() / "bin/conda"


def stop(process):
    if process and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            return
        try:
            process.wait(timeout=20)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def prepare_resumed_plan(run, plan):
    """Reject mixing checkpoints/protocols before trusting completed scenes."""
    existing = run / "evaluation_plan.json"
    if not existing.is_file():
        if any(run.glob("*/eval_logs/*.log")):
            raise ValueError("Existing evaluation logs have no fixed-checkpoint plan; use a new output directory.")
        return None
    previous = json.loads(existing.read_text())
    legacy_defaults = {"policy_rng": "legacy", "policy_rng_protocol": "process_global_torch_legacy",
                       "expert_cache_mode": "off", "expert_cache_identity": None}
    for key in ("checkpoint", "precision", "trials", "tasks", "settings", "inference_backend", "use_length",
                "num_steps", "compile", "action_smoothing", "expert_seed_policy", "policy_rng",
                "expert_cache_mode", "expert_cache_identity", "policy_rng_protocol"):
        if previous.get(key, legacy_defaults.get(key)) != plan.get(key, legacy_defaults.get(key)):
            raise ValueError(f"Evaluation resume mismatch for {key}; use the same checkpoint and protocol.")
    if previous.get("checkpoint_identity") is None and any(run.glob("*/eval_logs/*.log")):
        raise ValueError("Existing evaluation logs lack an attested checkpoint identity; use a new output directory.")
    return previous.get("checkpoint_identity")


def archive_previous_attempt(run, paths):
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    archive = run / "previous_attempts" / f"{time.time_ns()}-{os.getpid()}"
    for path in existing:
        destination = archive / path.relative_to(run)
        destination.parent.mkdir(parents=True, exist_ok=True)
        path.replace(destination)


def parse_args(argv=None):
    initial = argparse.ArgumentParser(add_help=False)
    initial.add_argument("--config", type=Path, default=ROOT / "configs/eval.yaml")
    config_args, _ = initial.parse_known_args(argv)
    profile = yaml.safe_load(config_args.config.read_text())
    allowed = {"backend", "precision", "use_length", "num_steps", "compile", "action_smoothing",
               "sim_workers", "policy_rng", "expert_cache", "expert_cache_mode"}
    if not isinstance(profile, dict) or set(profile) - allowed:
        raise ValueError("Unknown evaluation YAML fields; expected inference profile settings.")
    if "compile" in profile and type(profile["compile"]) is not bool:
        raise ValueError("Evaluation compile must be a YAML boolean.")
    for key in ("num_steps", "use_length", "sim_workers"):
        if key in profile and type(profile[key]) is not int:
            raise ValueError(f"Evaluation {key} must be a YAML integer.")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=config_args.config)
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
    parser.add_argument("--num-steps", type=int, default=10)
    parser.add_argument("--compile", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--action-smoothing", choices=["none"], default="none")
    parser.add_argument("--sim-workers", type=int, default=1, help="Independent task/setting clients sharing this server")
    parser.add_argument("--policy-rng", choices=["legacy", "scene"], default=None)
    parser.add_argument("--expert-cache", type=Path)
    parser.add_argument("--expert-cache-mode", choices=["off", "record", "replay"], default="off")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--backend", choices=["native", "streamed"], default="native")
    parser.add_argument("--native-server", dest="backend", action="store_const", const="native", help="Alias for --backend native")
    parser.add_argument("--server-timeout", type=int, default=1800)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--skip-completed", action="store_true", help="Resume this same fixed protocol, retaining completed tasks and archiving failed attempts")
    parser.add_argument("--team-id", default="replace_with_your_team_id")
    parser.set_defaults(**profile)
    args = parser.parse_args(argv)
    if args.backend not in {"native", "streamed"} or args.precision not in {"fp32", "bf16"}:
        parser.error("Invalid backend or precision in evaluation config")
    if args.action_smoothing != "none" or args.num_steps <= 0 or not 1 <= args.use_length <= 50:
        parser.error("Expected positive num_steps, use_length 1..50, action_smoothing none")
    if not 1 <= args.sim_workers <= 16:
        parser.error("sim_workers must be 1..16")
    args.policy_rng = args.policy_rng or ("scene" if args.sim_workers > 1 else "legacy")
    if args.policy_rng not in {"scene", "legacy"} or (args.sim_workers > 1 and args.policy_rng != "scene"):
        parser.error("Multiple simulators require policy_rng: scene")
    if args.expert_cache_mode not in {"off", "record", "replay"}:
        parser.error("Unknown expert_cache_mode")
    if args.expert_cache_mode != "off" and args.expert_cache is None:
        parser.error("record/replay requires --expert-cache")
    return args


def main():
    args = parse_args()
    if args.smoke:
        args.tasks, args.trials, args.precision = "adjust_bottle", 2, "bf16"
    tasks = parse_tasks(args.tasks)
    settings = list(INSTRUCTIONS) if args.setting == "both" else [args.setting]
    if not 1 <= args.trials <= 100:
        raise ValueError("--trials must be 1..100")
    if args.precision == "bf16":
        print("BF16 is an approximate pipeline check; official published validation uses FP32 and roughly 32 GB with simulation.")
    backend = "native" if args.backend == "native" else "streamed_sdpa_reference_moe"
    if args.backend != "native":
        print("The streamed server uses SDPA/reference MoE kernels; equivalence to the official FP32 server remains unverified.")
    run = args.output or ROOT / "runs/evaluation" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run = run.resolve()
    run.mkdir(parents=True, exist_ok=True)
    if args.aggregate_only:
        previous = json.loads((run / "evaluation_plan.json").read_text())
        profile = {"precision": args.precision, "inference_backend": backend, "use_length": args.use_length,
                   "num_steps": args.num_steps, "compile": args.compile, "action_smoothing": args.action_smoothing,
                   "expert_seed_policy": EXPERT_SEED_POLICY, "trials": args.trials, "tasks": tasks, "settings": settings,
                   "policy_rng": args.policy_rng, "expert_cache_mode": args.expert_cache_mode,
                   "policy_rng_protocol": POLICY_RNG if args.policy_rng == "scene" else "process_global_torch_legacy",
                   "expert_cache_identity": previous.get("expert_cache_identity"),
                   "checkpoint": str(args.checkpoint.resolve())}
        prepare_resumed_plan(run, profile)
        if base_checkpoint_identity(args.checkpoint.resolve()) != previous.get("checkpoint_identity"):
            raise ValueError("Aggregate checkpoint identity differs from the saved plan.")
        report = aggregate_results(run, tasks, settings, args.precision, args.trials, backend, use_length=args.use_length)
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
    client = create_eval_runtime(vendor, robotwin, run / "runtime", args.trials,
                                 scene_rng=args.policy_rng == "scene", expert_cache=args.expert_cache_mode != "off", timing=True)
    shared_environment = dict(os.environ, CUDA_VISIBLE_DEVICES=args.gpu, QWEN3VL_PATH=str(args.qwen.resolve()), PYTHONNOUSERSITE="1",
                              PYTHONUNBUFFERED="1", SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0", TOKENIZERS_PARALLELISM="false")
    shared_environment["PYTHONPATH"] = os.pathsep.join([str(run / "runtime"), str(vendor), str(robotwin), str(robotwin / "description/utils")])
    if args.sim_workers > 1 and not args.dry_run and not shared_environment.get("LRVLA_RENDER_PCI"):
        from lrvla.parallel_evaluation import assigned_devices
        device, = assigned_devices(args.gpu)
        args.gpu = device["uuid"]
        shared_environment.update(CUDA_VISIBLE_DEVICES=args.gpu, LRVLA_RENDER_PCI=device["pci"])
    cache_identity = None
    if args.expert_cache_mode != "off":
        from lrvla.scene_bank import source_identity
        cache_identity = {**source_identity(robotwin), "expert_seed_policy": EXPERT_SEED_POLICY,
                          "upstream_client_sha256": hashlib.sha256((vendor / "experiment/robotwin/eval_policy_client_lingbotvla.py").read_bytes()).hexdigest()}
        shared_environment.update(LRVLA_EXPERT_CACHE=str(args.expert_cache.resolve()),
                                  LRVLA_EXPERT_CACHE_MODE=args.expert_cache_mode,
                                  LRVLA_EXPERT_CACHE_IDENTITY=json.dumps(cache_identity))
    env = dict(shared_environment)
    env["PATH"] = os.pathsep.join(filter(None, [str(infer_python.parent), shared_environment.get("PATH", "")]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(conda_root / "envs" / args.inference_env / "lib"), shared_environment.get("LD_LIBRARY_PATH", "")]))
    sim_environment = dict(shared_environment)
    sim_environment["PATH"] = os.pathsep.join(filter(None, [str(sim_python.parent), shared_environment.get("PATH", "")]))
    sim_environment["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(conda_root / "envs" / args.sim_env / "lib"), shared_environment.get("LD_LIBRARY_PATH", "")]))
    if args.sim_workers > 1:
        sim_environment.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    server_entry = [str(ROOT / "scripts/serve_native.py" if args.backend == "native" else ROOT / "scripts/serve_streamed.py")]
    infer = [str(infer_python), *server_entry,
             "--model_path", str(checkpoint), "--use_length", str(args.use_length), "--use_bf16", str(args.precision == "bf16"),
             "--use_fp32", str(args.precision == "fp32"), "--use_compile", str(args.compile), "--port", str(args.port),
             "--num_steps", str(args.num_steps)]
    if args.policy_rng == "scene":
        infer.extend(["--policy-rng", "scene"])
    commands = []
    for setting in settings:
        for task in tasks:
            command = [str(sim_python), "-u", str(client),
                       "--config", "policy/pi0/deploy_policy.yml", "--overrides", "--task_name", task, "--task_config", setting,
                       "--ckpt_setting", "cotrain", "--train_config_name", "0", "--seed", "0", "--policy_name", "pi0",
                       "--port", str(args.port), "--robo_name", "robotwin", "--instruction_type", INSTRUCTIONS[setting],
                       "--eval_video_log", str(args.video), "--output_dir", str(run / setting / "eval_results")]
            commands.append({"setting": setting, "task": task, "command": command})
    limits_file = robotwin / "task_config/_eval_step_limit.yml"
    limits = yaml.safe_load(limits_file.read_text()) if limits_file.is_file() else {}
    # Start expensive task/setting pairs together to reduce the final tail.
    commands.sort(key=lambda item: (-limits.get(item["task"], 0), item["task"], item["setting"]))
    plan = {"checkpoint": str(checkpoint), "model_run": str(model_run), "precision": args.precision, "trials": args.trials,
            "inference_backend": backend, "use_length": args.use_length,
            "num_steps": args.num_steps, "compile": args.compile, "action_smoothing": args.action_smoothing,
            "policy_rng": args.policy_rng, "sim_workers": args.sim_workers,
            "policy_rng_protocol": POLICY_RNG if args.policy_rng == "scene" else "process_global_torch_legacy",
            "expert_cache_mode": args.expert_cache_mode, "expert_cache_identity": cache_identity,
            "expert_cache_replay_verified": False,
            "tasks": tasks, "settings": settings, "inference_command": infer, "evaluation_commands": commands,
            "instruction_types": INSTRUCTIONS, "expert_starts_per_scene": 5,
            "expert_seed_policy": EXPERT_SEED_POLICY, "fixed_checkpoint": True,
            "full_protocol": (len(tasks) == 50 and args.trials == 100 and len(settings) == 2
                              and args.precision == "fp32" and backend == "native" and args.use_length == 50
                              and args.num_steps == 10 and not args.compile and args.action_smoothing == "none"),
            "schema": "official schema_version=1; precision and protocol recorded in sidecar"}
    prior_identity = prepare_resumed_plan(run, plan) if args.skip_completed else None
    if prior_identity is not None:
        plan["checkpoint_identity"] = prior_identity
    (run / "evaluation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    if args.dry_run:
        print(f"Dry-run plan: {run / 'evaluation_plan.json'}")
        return
    if args.skip_completed:
        commands = [item for item in commands if not evaluation_task_complete(run, item["setting"], item["task"], args.trials)]
        if not commands:
            if base_checkpoint_identity(checkpoint) != prior_identity:
                raise ValueError("Evaluation checkpoint bytes changed after the saved completed scenes.")
            report = aggregate_results(run, tasks, settings, args.precision, args.trials, backend, use_length=args.use_length)
            export_official_results(report, run / "official_results.json", args.team_id, ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json")
            print(f"All requested scenes already completed: {run / 'results.json'}")
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
    if sim_environment.get("LRVLA_RENDER_PCI"):
        from lrvla.parallel_evaluation import inject_binding
        RENDER_CODE = inject_binding(RENDER_CODE)
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
    if prior_identity is not None and plan["checkpoint_identity"] != prior_identity:
        raise ValueError("Evaluation checkpoint bytes changed; use a new output directory.")
    (run / "evaluation_plan.json").write_text(json.dumps(plan, indent=2) + "\n")
    server = worker = sampler = None
    previous_sigterm = signal.getsignal(signal.SIGTERM)
    def terminate(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        archive_previous_attempt(run, [run / "inference.log"])
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
            if args.sim_workers > 1:
                from lrvla.evaluation_resources import ResourceSampler
                sampler = ResourceSampler(run / "resource_usage.jsonl", args.gpu)
                sampler.start()
            run_clients(commands, run=run, cwd=robotwin, environment=sim_environment,
                        workers=args.sim_workers, server=server, archive=archive_previous_attempt, stop=stop)
    finally:
        try:
            stop(worker)
            stop(server)
            if sampler:
                sampler.close()
            report = aggregate_results(run, tasks, settings, args.precision, args.trials, backend, use_length=args.use_length)
            export_official_results(report, run / "official_results.json", args.team_id, ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json")
        finally:
            signal.signal(signal.SIGTERM, previous_sigterm)
    report = json.loads((run / "results.json").read_text())
    print(f"Results: {run / 'results.json'} ({report['status']})")
    if not report["complete"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
