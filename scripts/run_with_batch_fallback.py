#!/usr/bin/env python3
"""Retry one-node training once with a smaller physical batch after CUDA OOM."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CUDA_OOM = re.compile(r"\bLRVLA_CUDA_OOM\b|\bCUDA(?:\s+error:)?\s+out\s+of\s+memory\b", re.IGNORECASE)
CONTROLLED_ARGUMENTS = {"--batch-size", "--global-batch-size", "--gradient-accumulation", "--output-dir", "--config", "--resume"}
TERMINATE_GRACE_SECONDS = 10


class LauncherInterrupted(Exception):
    def __init__(self, signum):
        self.signum = signum
        super().__init__(f"Launcher interrupted by signal {signum}")


def timestamp():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def save_report(path, report):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(report, indent=2) + "\n")
    temporary.replace(path)


def build_command(args, batch_size, run_dir):
    """Use the active Python environment and literal arguments, without a shell."""
    return [sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1",
            f"--nproc-per-node={args.gpus}", str(PROJECT_ROOT / "scripts/train.py"),
            "--config", str(args.config), "--batch-size", str(batch_size),
            "--global-batch-size", str(batch_size * args.gpus),
            "--gradient-accumulation", "1", "--output-dir", str(run_dir), *args.training_args]


def terminate_process_group(process):
    """Terminate only this launcher's child group, including surviving ranks."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=TERMINATE_GRACE_SECONDS)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=2)
    # The group leader may exit while a rank is still stuck in NCCL cleanup.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def run_attempt(args, global_batch, report, report_path):
    run_dir = args.output_dir / f"batch{global_batch}"
    log_path = args.output_dir / f"batch{global_batch}.launcher.log"
    batch_size = global_batch // args.gpus
    command = build_command(args, batch_size, run_dir)
    attempt = {"global_batch_size": global_batch, "batch_size_per_gpu": batch_size,
               "gradient_accumulation": 1, "config": str(args.config), "run_dir": str(run_dir),
               "config_sha256": hashlib.sha256(args.config.read_bytes()).hexdigest(),
               "config_overrides": {"global_batch_size": global_batch, "batch_size": batch_size,
                                    "gradient_accumulation": 1, "output_dir": str(run_dir)},
               "fallback": global_batch != args.global_batch_size,
               "log": str(log_path), "command": command, "started_at": timestamp(),
               "exit_code": None, "cuda_oom": False, "status": "running"}
    report["attempts"].append(attempt)
    save_report(report_path, report)
    process = None
    run_dir.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("w", buffering=1) as log:
            environment = dict(os.environ, PYTHONUNBUFFERED="1")
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, errors="replace", bufsize=1,
                                       start_new_session=True, env=environment)
            attempt["process_group"] = process.pid
            save_report(report_path, report)
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                log.write(line)
                if CUDA_OOM.search(line):
                    attempt.update(cuda_oom=True, oom_line=line.rstrip())
                    terminate_process_group(process)
                    break
            attempt["exit_code"] = process.wait()
            attempt["status"] = "cuda_oom" if attempt["cuda_oom"] else (
                "succeeded" if attempt["exit_code"] == 0 else "failed")
    except BaseException as error:
        if process is not None:
            terminate_process_group(process)
            attempt["exit_code"] = process.returncode
        attempt.update(status="interrupted" if isinstance(error, LauncherInterrupted) else "failed",
                       error=f"{type(error).__name__}: {error}")
        raise
    finally:
        if process is not None and process.stdout is not None:
            process.stdout.close()
        attempt["finished_at"] = timestamp()
        save_report(report_path, report)
    return attempt


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gpus", type=int, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--global-batch-size", type=int, required=True)
    parser.add_argument("--fallback-global-batch-size", type=int, required=True)
    parser.add_argument("training_args", nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    if args.gpus < 1:
        parser.error("--gpus must be positive")
    if not args.output_dir.is_absolute() or not args.config.is_absolute():
        parser.error("--output-dir and --config must be absolute Linux paths")
    if not args.config.is_file():
        parser.error("--config must name an existing YAML file")
    for batch in (args.global_batch_size, args.fallback_global_batch_size):
        if batch < 1 or batch % args.gpus:
            parser.error("Both global batch sizes must be positive multiples of --gpus")
    if args.fallback_global_batch_size >= args.global_batch_size:
        parser.error("The fallback global batch must be smaller than the initial batch")
    if args.training_args:
        if args.training_args[0] != "--":
            parser.error("Separate additional training arguments with --")
        args.training_args = args.training_args[1:]
    for argument in args.training_args:
        option = argument.split("=", 1)[0]
        # The training parser accepts unambiguous long-option abbreviations;
        # those must not bypass the launcher's physical batch or fresh start.
        if option.startswith("--") and any(controlled.startswith(option) for controlled in CONTROLLED_ARGUMENTS):
            parser.error(f"The launcher controls {option}; do not repeat it")
    try:
        config = yaml.safe_load(args.config.read_text())
    except (OSError, UnicodeError, yaml.YAMLError) as error:
        parser.error(f"Cannot read the training YAML: {error}")
    if not isinstance(config, dict):
        parser.error("The training YAML must contain a mapping")
    if config.get("resume"):
        parser.error("Batch fallback starts from the base checkpoint; remove resume from the training YAML")
    if (args.output_dir / "attempts.json").exists():
        parser.error("The output directory already contains attempts.json; choose a new run directory")
    for batch in (args.global_batch_size, args.fallback_global_batch_size):
        run_dir = args.output_dir / f"batch{batch}"
        if run_dir.exists() and (not run_dir.is_dir() or any(run_dir.iterdir())):
            parser.error(f"Attempt output must be empty: {run_dir}")
    return args


def main(argv=None):
    args = parse_args(argv)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "attempts.json"
    report = {"status": "running", "started_at": timestamp(), "config": str(args.config),
              "gpus": args.gpus, "gradient_accumulation": 1,
              "first_attempt_global_batch_size": args.global_batch_size,
              "fallback_global_batch_size": args.fallback_global_batch_size,
              "fallback_used": False, "attempts": []}
    old_handlers = {}
    def interrupt(signum, frame):
        raise LauncherInterrupted(signum)
    for signum in (signal.SIGINT, signal.SIGTERM):
        old_handlers[signum] = signal.signal(signum, interrupt)
    try:
        first = run_attempt(args, args.global_batch_size, report, report_path)
        if first["cuda_oom"]:
            report["fallback_used"] = True
            print(f"CUDA OOM at global batch {args.global_batch_size}; retrying once from the base "
                  f"checkpoint at global batch {args.fallback_global_batch_size}, gradient accumulation 1.",
                  flush=True)
            final = run_attempt(args, args.fallback_global_batch_size, report, report_path)
        else:
            final = first
        succeeded = final["exit_code"] == 0 and not final["cuda_oom"]
        report.update(status="succeeded" if succeeded else "failed",
                      actual_global_batch_size=final["global_batch_size"], run_dir=final["run_dir"])
        if succeeded:
            return 0
        exit_code = final["exit_code"] or 1
        return exit_code if exit_code > 0 else 128 - exit_code
    except LauncherInterrupted as error:
        report.update(status="interrupted", error=str(error))
        return 128 + error.signum
    except Exception as error:
        report.update(status="failed", error=f"{type(error).__name__}: {error}")
        print(report["error"], file=sys.stderr, flush=True)
        return 1
    finally:
        for signum, old_handler in old_handlers.items():
            signal.signal(signum, old_handler)
        report["finished_at"] = timestamp()
        save_report(report_path, report)


if __name__ == "__main__":
    raise SystemExit(main())
