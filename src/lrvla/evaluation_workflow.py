"""Prepare balanced task groups for one model service per allocated GPU."""
from __future__ import annotations

import json
from pathlib import Path
import re

from .data import parse_tasks
from .evaluation import EXPERT_SEED_POLICY
from .policy_sessions import POLICY_RNG


def balanced_groups(tasks, count, costs):
    groups, totals = [[] for _ in range(min(count, len(tasks)))], [0] * min(count, len(tasks))
    for task in sorted(tasks, key=lambda task: (-costs.get(task, 1), task)):
        index = min(range(len(groups)), key=lambda index: (totals[index], index))
        groups[index].append(task)
        totals[index] += costs.get(task, 1)
    return groups


def prepare_workflow(config, checkpoint, output, gpu_count, *, overrides=None, root=None):
    import yaml
    cfg = yaml.safe_load(Path(config).read_text())
    required = {"tasks", "settings", "trials", "backend", "precision", "use_length", "num_steps", "compile",
                "action_smoothing", "sim_workers", "policy_rng"}
    if not isinstance(cfg, dict) or not required <= set(cfg) or set(cfg) - required - {"expert_cache", "expert_cache_mode"}:
        raise ValueError("Unexpected parallel evaluation YAML fields.")
    cfg.update({k: v for k, v in (overrides or {}).items() if v is not None})
    tasks = parse_tasks(",".join(cfg["tasks"]) if isinstance(cfg["tasks"], list) else cfg["tasks"])
    if type(gpu_count) is not int or gpu_count < 1:
        raise ValueError("GPU count must be positive.")
    if type(cfg["sim_workers"]) is not int or not 1 <= cfg["sim_workers"] <= 16:
        raise ValueError("sim_workers must be 1..16.")
    if cfg["sim_workers"] > 1 and cfg["policy_rng"] != "scene":
        raise ValueError("Multiple simulators require scene policy_rng.")
    if (type(cfg["trials"]) is not int or not 1 <= cfg["trials"] <= 100
            or cfg["settings"] not in [["demo_clean"], ["demo_randomized"], ["demo_clean", "demo_randomized"]]
            or cfg["backend"] not in {"native", "streamed"} or cfg["precision"] not in {"fp32", "bf16"}
            or type(cfg["compile"]) is not bool or cfg["action_smoothing"] != "none"
            or type(cfg["use_length"]) is not int or not 1 <= cfg["use_length"] <= 50
            or type(cfg["num_steps"]) is not int or cfg["num_steps"] <= 0 or cfg["policy_rng"] not in {"legacy", "scene"}):
        raise ValueError("Invalid fixed evaluation profile.")
    mode = cfg.get("expert_cache_mode", "off")
    if mode not in {"off", "record", "replay"} or (mode != "off" and not cfg.get("expert_cache")):
        raise ValueError("record/replay needs an expert_cache directory.")
    root = Path(root or Path(__file__).resolve().parents[2])
    limits = root / "vendor/RoboTwin/task_config/_eval_step_limit.yml"
    costs = {task: int(cap) for task, cap in re.findall(r"^([a-z0-9_]+):\s*(\d+)\s*$", limits.read_text(), re.M)} if limits.exists() else {}
    output, checkpoint = Path(output).resolve(), Path(checkpoint).resolve()
    plan = {"schema": "lrvla-parallel-evaluation-v1", "run_dir": str(output), "checkpoint": str(checkpoint),
            "tasks": tasks, "settings": cfg["settings"], "trials": cfg["trials"],
            "precision": cfg["precision"], "inference_backend": cfg["backend"],
            "use_length": cfg["use_length"], "num_steps": cfg["num_steps"], "compile": cfg["compile"],
            "action_smoothing": cfg["action_smoothing"], "sim_workers": cfg["sim_workers"],
            "policy_rng": cfg["policy_rng"], "expert_seed_policy": EXPERT_SEED_POLICY,
            "policy_rng_protocol": POLICY_RNG if cfg["policy_rng"] == "scene" else "process_global_torch_legacy",
            "expert_cache_mode": mode, "expert_cache": str(Path(cfg["expert_cache"]).resolve()) if cfg.get("expert_cache") else None,
            "shards": [{"index": i, "tasks": members, "output": str(output / f"group-{i:03d}")}
                       for i, members in enumerate(balanced_groups(tasks, gpu_count, costs), 1)]}
    output.mkdir(parents=True, exist_ok=True)
    path = output / "workflow.json"
    if path.exists() and json.loads(path.read_text()) != plan:
        raise ValueError("Existing workflow differs or is stopped; choose a new output directory.")
    path.write_text(json.dumps(plan, indent=2) + "\n")
    return path


def aggregate_workflow(workflow):
    from .evaluation import aggregate_results
    mappings, identity, profile = {}, None, None
    for shard in workflow["shards"]:
        directory = Path(shard["output"])
        for task in shard["tasks"]:
            if task in mappings:
                raise ValueError("Repeated task in evaluation workflow.")
            mappings[task] = directory
        plan_file = directory / "evaluation_plan.json"
        if not plan_file.exists():
            if any(directory.glob("*/eval_logs/*.log")):
                raise ValueError("Simulator logs have no attested evaluation plan.")
            continue
        plan = json.loads(plan_file.read_text())
        for key in ("checkpoint", "precision", "trials", "settings", "use_length", "num_steps", "compile",
                    "action_smoothing", "expert_seed_policy", "policy_rng", "policy_rng_protocol", "sim_workers", "expert_cache_mode"):
            if plan.get(key) != workflow.get(key):
                raise ValueError(f"Parallel shard differs for {key}.")
        expected_backend = {"streamed": "streamed_sdpa_reference_moe"}.get(workflow["inference_backend"], workflow["inference_backend"])
        if plan.get("tasks") != shard["tasks"] or plan.get("inference_backend") != expected_backend:
            raise ValueError("Parallel shard has another task group or backend.")
        current = plan.get("checkpoint_identity")
        if current is None and any(directory.glob("*/eval_logs/*.log")):
            raise ValueError("Simulator logs lack checkpoint identity.")
        if current is not None:
            if identity is not None and identity != current:
                raise ValueError("Parallel shards evaluated different checkpoint bytes.")
            identity = current
        if profile and profile.get("expert_cache_identity") != plan.get("expert_cache_identity"):
            raise ValueError("Parallel shards used different scene cache sources.")
        profile = plan
    if set(mappings) != set(workflow["tasks"]):
        raise ValueError("Parallel workflow task coverage differs.")
    root = Path(workflow["run_dir"])
    root.mkdir(parents=True, exist_ok=True)
    backend = {"streamed": "streamed_sdpa_reference_moe"}.get(workflow["inference_backend"], workflow["inference_backend"])
    report = aggregate_results(root, workflow["tasks"], workflow["settings"], workflow["precision"],
                               workflow["trials"], backend, task_runs=mappings, use_length=workflow["use_length"],
                               expert_seed_policy=workflow["expert_seed_policy"], inference_profile=profile or workflow)
    report.update(checkpoint=workflow["checkpoint"], checkpoint_identity=identity)
    (root / "results.json").write_text(json.dumps(report, indent=2) + "\n")
    return report
