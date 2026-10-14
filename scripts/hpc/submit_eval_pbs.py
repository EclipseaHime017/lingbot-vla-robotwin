#!/usr/bin/env python3
"""Prepare task-sharded native FP32 RoboTwin evaluation jobs for PBS."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from lrvla.data import parse_tasks
from lrvla.evaluation import INSTRUCTIONS
from submit_pbs import absolute_path, dependency, positive_int, safe_token, walltime


def split_tasks(tasks, tasks_per_shard):
    if not tasks or tasks_per_shard <= 0 or len(tasks) != len(set(tasks)):
        raise ValueError("Shard size must be positive and tasks must be unique.")
    return [tasks[start:start + tasks_per_shard] for start in range(0, len(tasks), tasks_per_shard)]


def job_script(args, shard, log_dir):
    command = shlex.join([
        str(args.workdir / "scripts/evaluate.py"), "--checkpoint", str(args.checkpoint),
        "--tasks", ",".join(shard["tasks"]), "--setting", "both", "--trials", "100",
        "--precision", "fp32", "--native-server", "--use-length", str(args.use_length),
        "--output", shard["output"], "--skip-completed",
    ])
    port = str(args.port) if args.port is not None else '$((20000 + ${PBS_JOBID%%.*} % 30000))'
    return f"""#!/bin/bash
#PBS -N {shard['job_name']}
#PBS -P {args.project}
#PBS -q normal
#PBS -l select=1:ncpus=16:mem=110gb:ngpus=1
#PBS -l walltime={args.walltime}
#PBS -j oe
#PBS -m n
set -euo pipefail
umask 077
: "${{PBS_JOBID:?Run through PBS, not on a login node}}"
: "${{CUDA_VISIBLE_DEVICES:?The PBS GPU assignment must be preserved}}"
if [[ "$CUDA_VISIBLE_DEVICES" == *,* ]]; then
    printf 'This evaluation shard requires exactly one PBS GPU.\\n' >&2
    exit 2
fi
mkdir -p {shlex.quote(str(log_dir))}
exec >> {shlex.quote(str(log_dir))}/"${{PBS_JOBID}}.log" 2>&1
printf 'PBS_JOBID=%s CUDA_VISIBLE_DEVICES=%s\\n' "$PBS_JOBID" "$CUDA_VISIBLE_DEVICES"
hostname
ulimit -c 0
export CONDA_ROOT={shlex.quote(str(args.conda_root))}
export PYTHONUNBUFFERED=1
export PYTHONNOUSERSITE=1
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export MPLBACKEND=Agg
cd {shlex.quote(str(args.workdir))}
source scripts/activate.sh lingbot-vla
LRVLA_EVAL_PORT={port}
exec "${{CONDA_PREFIX}}/bin/python" {command} --gpu "$CUDA_VISIBLE_DEVICES" --port "$LRVLA_EVAL_PORT"
"""


def aggregate_script(args, log_dir):
    command = shlex.join([
        str(args.workdir / "scripts/hpc/aggregate_eval_shards.py"),
        "--workflow", str(args.run_dir / "evaluation_workflow.json"),
    ])
    return f"""#!/bin/bash
#PBS -N {args.job_name}agg
#PBS -P {args.project}
#PBS -q normal
#PBS -l select=1:ncpus=4:mem=16gb
#PBS -l walltime=00:20:00
#PBS -j oe
#PBS -m n
set -euo pipefail
umask 077
: "${{PBS_JOBID:?Run through PBS, not on a login node}}"
mkdir -p {shlex.quote(str(log_dir))}
exec >> {shlex.quote(str(log_dir))}/"${{PBS_JOBID}}.log" 2>&1
cd {shlex.quote(str(args.workdir))}
exec {shlex.quote(str(args.conda_root / 'bin/python'))} {command}
"""


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project", required=True, type=safe_token)
    parser.add_argument("--workdir", required=True, type=absolute_path)
    parser.add_argument("--conda-root", required=True, type=absolute_path)
    parser.add_argument("--checkpoint", required=True, type=absolute_path)
    parser.add_argument("--run-dir", required=True, type=absolute_path)
    parser.add_argument("--output-dir", type=absolute_path)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--tasks-per-shard", type=positive_int, default=5)
    parser.add_argument("--walltime", type=walltime, default="24:00:00")
    parser.add_argument("--dependency", type=dependency)
    parser.add_argument("--job-name", type=safe_token, default="lrvlaeval")
    parser.add_argument("--use-length", type=positive_int, default=50)
    parser.add_argument("--port", type=positive_int, help="Fixed inference port; defaults to a port derived from PBS_JOBID on shared nodes.")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    # evaluate.py resolves symlinks (notably HOME/runs -> scratch); use the
    # same canonical paths for fixed-checkpoint and shard-plan comparisons.
    args.checkpoint = args.checkpoint.resolve()
    args.run_dir = args.run_dir.resolve()
    duration = sum(value * multiplier for value, multiplier in zip(map(int, args.walltime.split(":")), (3600, 60, 1)))
    if not 2 * 3600 < duration <= 24 * 3600:
        parser.error("A long single-GPU normal job must request over 2 hours and at most 24 hours (g1 routing).")
    if len(args.job_name) > 12:
        parser.error("--job-name must be at most 12 characters, leaving room for shard/aggregate suffixes.")
    if args.port is not None and args.port > 65535:
        parser.error("--port must be at most 65535.")
    tasks = parse_tasks(args.tasks)
    output = args.output_dir or args.run_dir / "pbs"
    log_dir = output / "logs"
    args.run_dir.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    shards = []
    for index, members in enumerate(split_tasks(tasks, args.tasks_per_shard), 1):
        job_name = f"{args.job_name}{index:03d}"
        shard = {"index": index, "tasks": members, "job_name": job_name,
                 "output": str(args.run_dir / "shards" / f"shard-{index:03d}"),
                 "script": str(output / f"{job_name}.pbs")}
        script = Path(shard["script"])
        command = ["qsub", "-mn"]
        if args.dependency:
            command.extend(["-W", "depend=" + args.dependency])
        command.extend(["-o", str(log_dir / f"{job_name}.scheduler.log"), str(script)])
        shard["qsub_argv"] = command
        shards.append(shard)
    workflow = {
        "schema": "lrvla-pbs-evaluation-v1", "run_dir": str(args.run_dir),
        "checkpoint": str(args.checkpoint), "precision": "fp32", "inference_backend": "native",
        "trials": 100, "settings": list(INSTRUCTIONS), "use_length": args.use_length,
        "tasks": tasks, "tasks_per_shard": args.tasks_per_shard, "shards": shards,
    }
    manifest = args.run_dir / "evaluation_workflow.json"
    if manifest.exists():
        previous = json.loads(manifest.read_text())
        identity_keys = ("schema", "run_dir", "checkpoint", "precision", "inference_backend", "trials",
                         "settings", "use_length", "tasks", "tasks_per_shard")
        if any(previous.get(key) != workflow[key] for key in identity_keys):
            parser.error("Existing workflow uses another checkpoint, protocol or partition; choose a new --run-dir.")
        if len(previous["shards"]) != len(shards):
            parser.error("Existing workflow has a different shard count; choose a new --run-dir.")
        for shard, old in zip(shards, previous["shards"]):
            if old.get("tasks") != shard["tasks"] or old.get("output") != shard["output"]:
                parser.error("Existing workflow has different shard contents; choose a new --run-dir.")
            if "job_id" in old:
                shard["job_id"] = old["job_id"]
        if "aggregate_job_id" in previous:
            workflow["aggregate_job_id"] = previous["aggregate_job_id"]
        if "aggregate_qsub_argv" in previous:
            workflow["aggregate_qsub_argv"] = previous["aggregate_qsub_argv"]
    for shard in shards:
        script = Path(shard["script"])
        script.write_text(job_script(args, shard, log_dir))
        script.chmod(0o600)
    aggregate = output / f"{args.job_name}agg.pbs"
    aggregate.write_text(aggregate_script(args, log_dir))
    aggregate.chmod(0o600)
    workflow["aggregate_script"] = str(aggregate)
    manifest.write_text(json.dumps(workflow, indent=2) + "\n")
    if args.submit:
        for shard in shards:
            if "job_id" in shard:
                continue
            completed = subprocess.run(shard["qsub_argv"], capture_output=True, text=True, check=False)
            if completed.returncode:
                print(completed.stderr or completed.stdout, file=sys.stderr)
                return 1
            shard["job_id"] = completed.stdout.strip()
            manifest.write_text(json.dumps(workflow, indent=2) + "\n")
        if "aggregate_job_id" not in workflow:
            command = ["qsub", "-mn", "-W", "depend=afterany:" + ":".join(shard["job_id"] for shard in shards),
                       "-o", str(log_dir / f"{args.job_name}agg.scheduler.log"), str(aggregate)]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode:
                print(completed.stderr or completed.stdout, file=sys.stderr)
                return 1
            workflow["aggregate_job_id"] = completed.stdout.strip()
            workflow["aggregate_qsub_argv"] = command
            manifest.write_text(json.dumps(workflow, indent=2) + "\n")
    print(json.dumps(workflow, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
