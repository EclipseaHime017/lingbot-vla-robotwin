#!/usr/bin/env python3
"""Generate a one-node NSCC GPU PBS job; submit only with the explicit --submit flag."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import shlex
import subprocess
import sys


def safe_token(value: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", value):
        raise argparse.ArgumentTypeError("Use a PBS token containing letters, digits, '.', '_' or '-'.")
    return value


def absolute_path(value: str) -> Path:
    path = Path(value)
    if not path.is_absolute() or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise argparse.ArgumentTypeError("Use an absolute Linux path without control characters.")
    return path


def config_path(value: str) -> Path:
    if not value or any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise argparse.ArgumentTypeError("Use a nonempty path without control characters.")
    return Path(value)


def positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("Use a positive integer.") from error
    if number <= 0:
        raise argparse.ArgumentTypeError("Use a positive integer.")
    return number


def expert_layers(value: str) -> int:
    number = int(value)
    if number != -1 and not 1 <= number <= 36:
        raise argparse.ArgumentTypeError("Use -1 or a layer count between 1 and 36.")
    return number


def walltime(value: str) -> str:
    if not re.fullmatch(r"\d{2,3}:[0-5]\d:[0-5]\d", value):
        raise argparse.ArgumentTypeError("Use HH:MM:SS with valid minutes and seconds.")
    hours, minutes, seconds = map(int, value.split(":"))
    if not 0 < hours * 3600 + minutes * 60 + seconds <= 120 * 3600:
        raise argparse.ArgumentTypeError("Walltime must be positive and at most 120 hours.")
    return value


def dependency(value: str) -> str:
    """Allow only afterok dependencies on ordinary PBS job identifiers."""
    identifiers = value.removeprefix("afterok:").split(":")
    if not identifiers or any(
        not re.fullmatch(r"[0-9]+(?:\.[A-Za-z0-9][A-Za-z0-9_.-]{0,127})?", item)
        for item in identifiers
    ):
        raise argparse.ArgumentTypeError("Use a PBS job ID or afterok:<job-id>[:<job-id>...].")
    return "afterok:" + ":".join(identifiers)


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--project", required=True, type=safe_token, help="Authorized NSCC project ID.")
    result.add_argument("--queue", required=True, choices=("normal", "ai"), help="Queue your project can access.")
    result.add_argument("--workdir", required=True, type=absolute_path, help="Repository path on the PBS system.")
    result.add_argument("--config", required=True, type=config_path, help="Training YAML, relative to workdir or absolute.")
    result.add_argument("--conda-root", required=True, type=absolute_path, help="Linux Miniconda/Miniforge prefix.")
    result.add_argument("--output-dir", type=absolute_path,
                        help="Persistent job scripts/log directory; defaults to workdir/artifacts/pbs.")
    result.add_argument("--job-name", type=safe_token, help="Defaults to lrvla, lrvla4 or lrvla8.")
    result.add_argument("--gpus", default=1, type=int, choices=(1, 4, 8), help="GPU count on a single node.")
    result.add_argument("--walltime", type=walltime, help="Defaults to 20 minutes on one GPU, 24 hours on four/eight.")
    result.add_argument("--dependency", type=dependency,
                        help="Wait for successful completion of one or more PBS jobs.")
    result.add_argument("--threads", default=4, type=positive_int, help="OMP/MKL threads per process.")
    result.add_argument("--allow-subset", action="store_true")
    result.add_argument("--max-steps", type=positive_int, help="Optimizer updates, including already restored updates.")
    result.add_argument("--batch-size", type=positive_int)
    result.add_argument("--gradient-accumulation", type=positive_int)
    result.add_argument("--global-batch-size", type=positive_int, help="Total samples per optimizer update.")
    result.add_argument("--batch-policy", choices=("fixed", "cuda-oom"),
                        help="fixed runs one batch size; cuda-oom retries once at the fallback batch. "
                             "Defaults to cuda-oom when --fallback-global-batch-size is supplied, otherwise fixed.")
    result.add_argument("--fallback-global-batch-size", type=positive_int,
                        help="On CUDA OOM, retry once with this smaller batch and no accumulation.")
    result.add_argument("--expert-last-n-layers", type=expert_layers)
    result.add_argument("--resume", type=config_path, help="Training checkpoint, relative to workdir or absolute.")
    result.add_argument("--train-output-dir", type=absolute_path, help="Override the YAML training output directory.")
    result.add_argument("--submit", action="store_true", help="Actually call qsub; otherwise only write the job script.")
    return result


def training_output_dir(args: argparse.Namespace) -> Path | None:
    if args.train_output_dir is not None:
        return args.train_output_dir
    if args.gpus > 1:
        return args.workdir / "runs" / f"{args.job_name}_{args.gpus}gpu"
    return None


def train_arguments(args: argparse.Namespace) -> list[str]:
    config = args.config if args.config.is_absolute() else args.workdir / args.config
    arguments = ["--config", str(config)]
    for option in ("max_steps", "batch_size", "gradient_accumulation", "global_batch_size", "expert_last_n_layers"):
        value = getattr(args, option)
        if value is not None:
            arguments.extend(["--" + option.replace("_", "-"), str(value)])
    if args.allow_subset:
        arguments.append("--allow-subset")
    if args.resume is not None:
        resume = args.resume if args.resume.is_absolute() else args.workdir / args.resume
        arguments.extend(["--resume", str(resume)])
    train_output = training_output_dir(args)
    if train_output is not None:
        arguments.extend(["--output-dir", str(train_output)])
    return arguments


def fallback_arguments(args: argparse.Namespace) -> list[str]:
    config = args.config if args.config.is_absolute() else args.workdir / args.config
    arguments = [
        str(args.workdir / "scripts" / "hpc" / "run_with_batch_fallback.py"),
        "--gpus", str(args.gpus), "--output-dir", str(training_output_dir(args)),
        "--config", str(config), "--global-batch-size", str(args.global_batch_size),
        "--fallback-global-batch-size", str(args.fallback_global_batch_size),
    ]
    extra_arguments = []
    for option in ("max_steps", "expert_last_n_layers"):
        value = getattr(args, option)
        if value is not None:
            extra_arguments.extend(["--" + option.replace("_", "-"), str(value)])
    if args.allow_subset:
        extra_arguments.append("--allow-subset")
    if extra_arguments:
        arguments.extend(["--", *extra_arguments])
    return arguments


def job_script(args: argparse.Namespace, output: Path) -> str:
    if args.fallback_global_batch_size is not None:
        command = '"${CONDA_PREFIX}/bin/python" ' + shlex.join(fallback_arguments(args))
    elif args.gpus == 1:
        command = shlex.join(["/bin/bash", str(args.workdir / "train.sh"), *train_arguments(args)])
    else:
        arguments = shlex.join([
            "--standalone", "--nnodes=1", f"--nproc-per-node={args.gpus}",
            str(args.workdir / "scripts" / "train.py"), *train_arguments(args),
        ])
        command = '"${CONDA_PREFIX}/bin/torchrun" ' + arguments
    gpu_count = "one" if args.gpus == 1 else str(args.gpus)
    return f"""#!/bin/bash
#PBS -N {args.job_name}
#PBS -P {args.project}
#PBS -q {args.queue}
#PBS -l select=1:ngpus={args.gpus}
#PBS -l walltime={args.walltime}
#PBS -j oe
#PBS -m n
set -euo pipefail
umask 077
: "${{PBS_JOBID:?Run this script through PBS, not on a login node}}"
LRVLA_JOB_LOG_DIR={shlex.quote(str(output / 'logs'))}
mkdir -p "$LRVLA_JOB_LOG_DIR"
exec >> "$LRVLA_JOB_LOG_DIR/${{PBS_JOBID}}.log" 2>&1
export CONDA_ROOT={shlex.quote(str(args.conda_root))}
export OMP_NUM_THREADS={args.threads}
export MKL_NUM_THREADS={args.threads}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
cd {shlex.quote(str(args.workdir))}
printf 'PBS_JOBID=%s\\n' "$PBS_JOBID"
printf 'batch_policy=%s\\n' {shlex.quote(args.batch_policy)}
printf 'CUDA_VISIBLE_DEVICES=%s\\n' "${{CUDA_VISIBLE_DEVICES-<unset>}}"
hostname
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi
else
    printf 'nvidia-smi unavailable; checking CUDA through Torch.\\n'
fi
source scripts/activate.sh lingbot-vla
"${{CONDA_PREFIX}}/bin/python" - <<'LRVLA_GPU_CHECK'
import json
import sys
import torch
available = torch.cuda.is_available()
count = torch.cuda.device_count() if available else 0
report = {{"python": sys.version, "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
          "cuda_available": available, "visible_gpu_count": count}}
if available:
    report.update(gpu=torch.cuda.get_device_name(0),
                  vram_gib=torch.cuda.get_device_properties(0).total_memory / 2**30,
                  capability=list(torch.cuda.get_device_capability(0)))
print(json.dumps(report), flush=True)
if count != {args.gpus}:
    raise RuntimeError("This training entrypoint requires exactly {gpu_count} PBS-assigned visible GPU(s).")
LRVLA_GPU_CHECK
exec {command}
"""


def main(argv: list[str] | None = None) -> int:
    argument_parser = parser()
    args = argument_parser.parse_args(argv)
    if args.walltime is None:
        args.walltime = "00:20:00" if args.gpus == 1 else "24:00:00"
    if args.job_name is None:
        args.job_name = "lrvla" if args.gpus == 1 else f"lrvla{args.gpus}"
    if len(args.job_name) > 15:
        argument_parser.error("--job-name must be at most 15 characters for portable PBS compatibility.")
    if args.batch_policy is None:
        args.batch_policy = "cuda-oom" if args.fallback_global_batch_size is not None else "fixed"
    if args.batch_policy == "fixed" and args.fallback_global_batch_size is not None:
        argument_parser.error("--batch-policy fixed cannot be combined with --fallback-global-batch-size.")
    if args.batch_policy == "cuda-oom" and args.fallback_global_batch_size is None:
        argument_parser.error("--batch-policy cuda-oom requires --fallback-global-batch-size.")
    if args.global_batch_size is not None:
        if args.gpus not in (4, 8):
            argument_parser.error("--global-batch-size requires a four- or eight-GPU job.")
        if args.global_batch_size % args.gpus:
            argument_parser.error("--global-batch-size must be divisible by --gpus.")
    if args.fallback_global_batch_size is not None:
        if args.gpus not in (4, 8) or args.global_batch_size is None:
            argument_parser.error("--fallback-global-batch-size requires four/eight GPUs and --global-batch-size.")
        if args.fallback_global_batch_size % args.gpus:
            argument_parser.error("--fallback-global-batch-size must be divisible by --gpus.")
        if args.fallback_global_batch_size >= args.global_batch_size:
            argument_parser.error("--fallback-global-batch-size must be smaller than --global-batch-size.")
        if args.batch_size is not None or args.gradient_accumulation is not None or args.resume is not None:
            argument_parser.error("Batch fallback cannot be combined with --batch-size, --gradient-accumulation or --resume.")
    if args.gpus == 8:
        if args.queue != "ai":
            argument_parser.error("A single eight-GPU node requires the ai queue; normal nodes have four GPUs.")
        hours, minutes, seconds = map(int, args.walltime.split(":"))
        duration = hours * 3600 + minutes * 60 + seconds
        if not 2 * 3600 < duration <= 24 * 3600:
            argument_parser.error("The eight-GPU ai pool requires walltime over 2 hours and at most 24 hours.")
    output = args.output_dir or args.workdir / "artifacts" / "pbs"
    try:
        output.mkdir(parents=True, exist_ok=True)
        (output / "logs").mkdir(exist_ok=True)
        script = output / f"{args.job_name}.pbs"
        script.write_text(job_script(args, output))
        script.chmod(0o600)
    except OSError as error:
        argument_parser.error(f"Cannot write persistent job directory: {error}")
    command = ["qsub", "-mn"]
    if args.dependency is not None:
        command.extend(["-W", "depend=" + args.dependency])
    command.extend(["-o", str(output / "logs" / f"{args.job_name}.scheduler.log"), str(script)])
    report = {"script": str(script), "qsub_argv": command, "submitted": False,
              "batch_policy": args.batch_policy}
    if args.submit:
        try:
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
        except OSError as error:
            print(f"Cannot execute qsub: {error}", file=sys.stderr)
            return 1
        if completed.returncode:
            print(completed.stderr or completed.stdout or "qsub failed.", end="\n", file=sys.stderr)
            return completed.returncode if completed.returncode > 0 else 128 - completed.returncode
        report.update(submitted=True, job_id=completed.stdout.strip())
    print(json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
