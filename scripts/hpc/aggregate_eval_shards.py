#!/usr/bin/env python3
"""Collect task-sharded results without treating missing scenes as complete."""
# ruff: noqa: E402
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import parse_tasks
from lrvla.evaluation import aggregate_results, export_official_results


def collect_workflow(workflow):
    if workflow.get("schema") != "lrvla-pbs-evaluation-v1":
        raise ValueError("Unsupported evaluation workflow schema.")
    if not workflow.get("tasks") or not workflow.get("shards"):
        raise ValueError("An evaluation workflow needs nonempty task shards.")
    tasks = parse_tasks(",".join(workflow["tasks"]))
    task_runs = {}
    identities = []
    for shard in workflow["shards"]:
        if not shard.get("tasks"):
            raise ValueError("An evaluation shard cannot have an empty task list.")
        members = parse_tasks(",".join(shard["tasks"]))
        run = Path(shard["output"])
        for task in members:
            if task in task_runs:
                raise ValueError(f"Evaluation workflow repeats task {task} across shards.")
            task_runs[task] = run
        plan_file = run / "evaluation_plan.json"
        if not plan_file.is_file():
            if any(run.glob("*/eval_logs/*.log")):
                raise ValueError(f"Evaluation logs lack the shard's attested plan: {run}")
            continue
        plan = json.loads(plan_file.read_text())
        for key in ("checkpoint", "precision", "trials", "settings", "inference_backend", "use_length"):
            if plan.get(key) != workflow.get(key):
                raise ValueError(f"Shard evaluation mismatch for {key}: {run}")
        if plan.get("tasks") != members:
            raise ValueError(f"Shard evaluation task list differs from its workflow: {run}")
        identity = plan.get("checkpoint_identity")
        if identity is None and any(run.glob("*/eval_logs/*.log")):
            raise ValueError(f"Shard logs lack an attested checkpoint identity: {run}")
        if identity is not None:
            identities.append(identity)
    if set(task_runs) != set(tasks):
        raise ValueError("Workflow shard task coverage differs from its requested tasks.")
    if identities and any(identity != identities[0] for identity in identities[1:]):
        raise ValueError("Shards evaluated different checkpoint bytes.")
    report = aggregate_results(Path(workflow["run_dir"]), tasks, workflow["settings"], workflow["precision"],
                               workflow["trials"], workflow["inference_backend"], task_runs=task_runs,
                               use_length=workflow["use_length"])
    report["checkpoint"] = workflow["checkpoint"]
    report["checkpoint_identity"] = identities[0] if identities else None
    report["evaluation_shards"] = len(workflow["shards"])
    (Path(workflow["run_dir"]) / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow", required=True, type=Path)
    parser.add_argument("--team-id", default="replace_with_your_team_id")
    args = parser.parse_args(argv)
    workflow = json.loads(args.workflow.read_text())
    report = collect_workflow(workflow)
    export_official_results(report, Path(workflow["run_dir"]) / "official_results.json", args.team_id,
                            ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json")
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
