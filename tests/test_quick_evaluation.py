"""Fixed 50 x 2 x 10 quick evaluation: planning, orchestration, collection and summary."""
import json
import math
from pathlib import Path

import pytest

from lrvla import quick_evaluation as quick
from lrvla.data import TASKS

LIMITS = {task: 400 + 100 * (index % 7) for index, task in enumerate(TASKS)}
DEVICES = [{"uuid": f"GPU-{name}", "pci": f"0000:0{index}:00.0", "name": "test", "total_mib": 40960, "used_mib": 0}
           for index, name in enumerate("ab", 1)]
FAKE_EVALUATOR = '''import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
for name in ("--config", "--checkpoint", "--tasks", "--setting", "--trials", "--precision", "--output", "--gpu",
             "--port", "--sim-workers", "--policy-rng"):
    parser.add_argument(name)
parser.add_argument("--skip-completed", action="store_true")
parser.add_argument("--dry-run", action="store_true")
args = parser.parse_args()
output, tasks = Path(args.output), args.tasks.split(",")
output.mkdir(parents=True, exist_ok=True)
with (output / "invocations.jsonl").open("a") as stream:
    stream.write(json.dumps({"gpu": args.gpu, "port": args.port, "render_pci": os.environ.get("LRVLA_RENDER_PCI"),
                             "sim_workers": args.sim_workers, "policy_rng": args.policy_rng}) + "\\n")
(output / "evaluation_plan.json").write_text(json.dumps({
    "precision": args.precision, "inference_backend": "native", "use_length": 50, "num_steps": 10, "compile": False,
    "action_smoothing": "none", "policy_rng": args.policy_rng, "policy_rng_protocol": "test",
    "expert_seed_policy": "bounded_advance_count_unstable_v2", "expert_cache_mode": "off", "trials": int(args.trials),
    "settings": ["demo_clean", "demo_randomized"], "tasks": tasks, "sim_workers": int(args.sim_workers),
    "checkpoint_identity": {"sha256": os.environ.get("FAKE_IDENTITY", "weights")}}))
if args.dry_run:
    sys.exit(0)
fail = os.environ.get("FAKE_FAIL_SHARD") == output.name and not (output / "failed-once").exists()
for setting, offset in (("demo_clean", 0), ("demo_randomized", 3)):
    for index, task in enumerate(tasks):
        logs = output / setting / "eval_logs"
        logs.mkdir(parents=True, exist_ok=True)
        successes = (len(task) + offset) % 11
        lines = [f"step: 5 / 400\\rLRVLA_EPISODE_METRICS=" + json.dumps({"expert_seconds": 2.0, "action_seconds": 6.0,
                                                                          "server_infer_seconds": 1.0, "total_seconds": 10.0})]
        lines += [f"Success rate: {min(scene, successes)}/{scene} => 0.0%, current seed: {scene}" for scene in range(1, 11)]
        (logs / f"{task}.log").write_text("\\n".join(lines) + "\\n")
        (logs / f"{task}.status.json").write_text(json.dumps({"exit_code": 0, "elapsed_seconds": 100.0}))
        if fail:
            (output / "failed-once").write_text("")
            (logs / f"{task}.status.json").write_text(json.dumps({"exit_code": 1}))
            sys.exit(1)
'''


def model_dir(root, step=7):
    run = root / "model"
    (run / f"checkpoints/global_step_{step}/hf_ckpt").mkdir(parents=True)
    (run / f"checkpoints/global_step_{step}/hf_ckpt/model-00001-of-00001.safetensors").write_bytes(b"")
    (run / "configs/robot_configs").mkdir(parents=True)
    (run / "configs/robot_configs/robotwin.yaml").write_text("{}\n")
    (run / "lingbotvla_cli.yaml").write_text("{}\n")
    return run


@pytest.fixture
def fake(tmp_path, monkeypatch):
    evaluator = tmp_path / "fake_evaluate.py"
    evaluator.write_text(FAKE_EVALUATOR)
    monkeypatch.setattr(quick, "step_limits", lambda robotwin: dict(LIMITS))
    monkeypatch.setattr(quick, "gpu_inventory", lambda selectors: [dict(device) for device in DEVICES
                                                                   if device["uuid"] in selectors.split(",")])
    monkeypatch.delenv("PBS_JOBID", raising=False)
    return evaluator


def test_protocol_is_fixed_at_50_tasks_2_settings_10_scenes(tmp_path):
    for override in (["--trials", "5"], ["--tasks", "adjust_bottle"], ["--setting", "demo_clean"]):
        with pytest.raises(SystemExit):
            quick.parse_args(["--checkpoint", str(tmp_path), *override])
    plan = quick.build_plan(tmp_path, (1, 1), 1, LIMITS, quick.ROOT / "configs/eval.yaml", "fp32")
    assert plan["trials"] == 10 and plan["settings"] == ["demo_clean", "demo_randomized"]
    assert sorted(plan["part_tasks"]) == sorted(TASKS) and plan["shards"][0]["tasks"] == sorted(TASKS)
    arguments = quick.evaluator_arguments(plan, plan["shards"][0], tmp_path, "GPU-a", 1234)
    assert arguments[arguments.index("--trials") + 1] == "10" and arguments[arguments.index("--setting") + 1] == "both"
    assert "--skip-completed" in arguments and arguments[arguments.index("--policy-rng") + 1] == "legacy"
    shared = quick.build_plan(tmp_path, (1, 1), 1, LIMITS, quick.ROOT / "configs/eval.yaml", "fp32", sims_per_gpu=3)
    arguments = quick.evaluator_arguments(shared, shared["shards"][0], tmp_path, "GPU-a", 1234)
    assert arguments[arguments.index("--sim-workers") + 1] == "3"
    assert arguments[arguments.index("--policy-rng") + 1] == "scene"


def test_parts_and_shards_cover_every_task_once_with_balanced_step_limits():
    for count in (1, 2, 3, 5):
        groups = quick.balanced_split(list(TASKS), count, LIMITS)
        assert groups == quick.balanced_split(list(reversed(TASKS)), count, LIMITS)
        assert sorted(task for group in groups for task in group) == sorted(TASKS)
        loads = [sum(LIMITS[task] for task in group) for group in groups]
        assert max(loads) - min(loads) <= max(LIMITS.values())
    assert quick.parse_part("2/3") == (2, 3)
    for value in ("0/2", "3/2", "1/51", "a/b", "1"):
        with pytest.raises(Exception):
            quick.parse_part(value)


def test_two_gpus_retry_bind_rendering_and_summarize_against_baseline(tmp_path, fake, monkeypatch):
    baseline = json.loads(quick.TEMPLATE.read_text())
    for label in ("clean", "randomized"):
        baseline["results"][label] = {task: {"attempts": 10, "successes": 5} for task in TASKS}
    (tmp_path / "baseline.json").write_text(json.dumps(baseline))
    monkeypatch.setenv("FAKE_FAIL_SHARD", "s1")
    output, model = tmp_path / "quick", model_dir(tmp_path)
    code = quick.main(["--checkpoint", str(model), "--gpus", "GPU-a,GPU-b", "--output", str(output),
                       "--progress-interval", "1", "--evaluate-script", str(fake),
                       "--baseline", str(tmp_path / "baseline.json")])
    assert code == 0
    plan = json.loads((output / "quick_plan.json").read_text())
    assert len(plan["shards"]) == 2 and plan["checkpoint"].endswith("global_step_7/hf_ckpt")
    calls = [json.loads(line) for shard in plan["shards"]
             for line in (output / "shards" / shard["name"] / "invocations.jsonl").read_text().splitlines()]
    assert {(call["gpu"], call["render_pci"]) for call in calls} == {(gpu["uuid"], gpu["pci"]) for gpu in DEVICES}
    assert len({call["port"] for call in calls}) == len(calls) == 3
    attempts = json.loads((output / "status.json").read_text())
    assert [row["exit_code"] for row in attempts if row["shard"] == "s1"] == [1, 0]
    results = json.loads((output / "results.json").read_text())
    assert results["complete"] and not results["full_50_task_benchmark"]
    official = json.loads((output / "official_results.json").read_text())
    assert all(row["attempts"] == 10 for label in ("clean", "randomized") for row in official["results"][label].values())
    summary = json.loads((output / "summary.json").read_text())
    expected = {setting: sum((len(task) + offset) % 11 for task in TASKS) / 500
                for setting, offset in (("demo_clean", 0), ("demo_randomized", 3))}
    assert summary["settings"]["demo_clean"]["rate"] == pytest.approx(expected["demo_clean"])
    assert summary["mean"]["rate"] == pytest.approx(sum(expected.values()) / 2)
    assert summary["baseline"]["matched_pairs"] == 100
    assert summary["baseline"]["difference"]["mean"]["delta"] == pytest.approx(sum(expected.values()) / 2 - 0.5)
    assert summary["time_per_scene"]["episodes"] == 100 and summary["time_per_scene"]["action_seconds"] == 6.0
    assert summary["policy_rng"] == ["legacy"]
    assert "两设定平均" in (output / "summary.md").read_text()
    # Rerunning a finished output on fewer GPUs keeps its layout and launches nothing.
    assert quick.main(["--checkpoint", str(model), "--gpus", "GPU-a", "--output", str(output),
                       "--evaluate-script", str(fake)]) == 0
    assert len(json.loads((output / "status.json").read_text())) == 3


def test_resume_reuses_the_saved_layout_and_rejects_changed_runs(tmp_path):
    config = quick.ROOT / "configs/eval.yaml"
    plan = quick.build_plan(tmp_path / "a", (1, 1), 2, LIMITS, config, "fp32")
    quick.write_json(tmp_path / "quick_plan.json", plan)
    resumed = quick.reuse_plan(tmp_path, quick.build_plan(tmp_path / "a", (1, 1), 1, LIMITS, config, "fp32"))
    assert resumed["shards"] == plan["shards"] and len(resumed["shards"]) == 2
    for changed in (quick.build_plan(tmp_path / "b", (1, 1), 2, LIMITS, config, "fp32"),
                    quick.build_plan(tmp_path / "a", (1, 1), 2, LIMITS, config, "bf16"),
                    quick.build_plan(tmp_path / "a", (1, 2), 2, LIMITS, config, "fp32"),
                    quick.build_plan(tmp_path / "a", (1, 1), 2, LIMITS, config, "fp32", sims_per_gpu=2)):
        with pytest.raises(ValueError):
            quick.reuse_plan(tmp_path, changed)
    shared = quick.build_plan(tmp_path / "a", (1, 1), 2, LIMITS, config, "fp32", sims_per_gpu=3)
    quick.write_json(tmp_path / "quick_plan.json", shared)
    fewer = quick.reuse_plan(tmp_path, quick.build_plan(tmp_path / "a", (1, 1), 2, LIMITS, config, "fp32"))
    assert fewer["policy_rng"] == "scene" and fewer["sims_per_gpu"] == 1


def test_collect_merges_machine_parts_and_rejects_mixed_weights(tmp_path, fake, monkeypatch):
    model = model_dir(tmp_path)
    outputs = [tmp_path / f"part{index}" for index in (1, 2)]
    for index, output in enumerate(outputs, 1):
        code = quick.main(["--checkpoint", str(model), "--part", f"{index}/2", "--output", str(output), "--gpus", "GPU-a",
                           "--evaluate-script", str(fake),
                           *(["--sims-per-gpu", "2", "--no-memory-check"] if index == 2 else [])])
        assert code == 1  # each part alone covers half of the 50 tasks
    parts = [set(json.loads((output / "quick_plan.json").read_text())["part_tasks"]) for output in outputs]
    assert not parts[0] & parts[1] and parts[0] | parts[1] == set(TASKS)
    merged = quick.collect(outputs, tmp_path / "merged")
    assert merged["complete"] and merged["scenes"] == 1000 and merged["policy_rng"] == ["legacy", "scene"]
    with pytest.raises(ValueError):
        quick.collect([outputs[0], outputs[0]], tmp_path / "duplicate")
    plan_file = outputs[1] / "shards/s0/evaluation_plan.json"
    plan_file.write_text(json.dumps({**json.loads(plan_file.read_text()), "checkpoint_identity": {"sha256": "other"}}))
    with pytest.raises(ValueError):
        quick.collect(outputs, tmp_path / "mixed")
    plan_file.write_text(json.dumps({**json.loads(plan_file.read_text()), "checkpoint_identity": {"sha256": "weights"},
                                     "use_length": 25}))
    with pytest.raises(ValueError):
        quick.collect(outputs, tmp_path / "profile")


def test_statistics_and_baseline_formats(tmp_path):
    counts = {task: (5, 10) for task in TASKS}
    part = quick.setting_summary(counts)
    assert part["rate"] == 0.5 and part["ci95"] == pytest.approx(1.96 * math.sqrt(50 * 2.5) / 500)
    mean = quick.mean_summary([part, part])
    assert mean["rate"] == 0.5 and mean["ci95"] == pytest.approx(part["ci95"] / math.sqrt(2))
    report = {"schema": "lrvla-evaluation-v1", "settings": {
        setting: {"tasks": [{"task": task, "successes": 3, "trials": 10} for task in TASKS]} for setting in quick.SETTINGS}}
    (tmp_path / "results.json").write_text(json.dumps(report))
    assert quick.load_counts(tmp_path)["demo_randomized"]["adjust_bottle"] == (3, 10)
    report["settings"]["demo_clean"]["tasks"][0]["successes"] = 11
    (tmp_path / "results.json").write_text(json.dumps(report))
    with pytest.raises(ValueError):
        quick.load_counts(tmp_path / "results.json")


def test_checkpoint_resolution_memory_and_cluster_guards(tmp_path, monkeypatch):
    run = model_dir(tmp_path)
    assert quick.resolve_checkpoint(run) == (run / "checkpoints/global_step_7/hf_ckpt").resolve()
    assert quick.default_output(quick.resolve_checkpoint(run), (2, 3), "bf16").name == "quick-model-step7-bf16-part2of3"
    (run / "checkpoints/global_step_8/hf_ckpt").mkdir(parents=True)
    with pytest.raises(FileNotFoundError):
        quick.resolve_checkpoint(run)
    (run / "configs/robot_configs/robotwin.yaml").unlink()
    with pytest.raises(FileNotFoundError):
        quick.resolve_checkpoint(run / "checkpoints/global_step_7/hf_ckpt")
    small = [{**DEVICES[0], "total_mib": 40960, "used_mib": 8 * 1024}]
    with pytest.raises(RuntimeError):
        quick.check_memory(small, "fp32")
    quick.check_memory(small, "bf16")
    quick.check_memory([DEVICES[0]], "fp32")
    with pytest.raises(RuntimeError):
        quick.check_memory([DEVICES[0]], "fp32", sims_per_gpu=2)
    monkeypatch.setenv("PBS_JOBID", "1.pbs")
    for extra in (["--gpus", "0,1"], ["--sims-per-gpu", "2"]):
        with pytest.raises(SystemExit):
            quick.parse_args(["--checkpoint", str(run), *extra])
    assert quick.parse_args(["--checkpoint", str(run)]).gpu_list == ["0"]
