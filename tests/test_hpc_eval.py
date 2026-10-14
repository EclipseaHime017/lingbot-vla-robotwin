"""Evaluation partitions conserve scenes and never merge incompatible runs."""
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

import pytest

from lrvla.data import TASKS
from lrvla.evaluation import INSTRUCTIONS, aggregate_results, evaluation_task_complete


ROOT = Path(__file__).resolve().parents[1]


def load_script(name, path):
    spec = importlib.util.spec_from_file_location(name, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


submit = load_script("lrvla_eval_pbs", "scripts/hpc/submit_eval_pbs.py")
aggregate = load_script("lrvla_eval_shards", "scripts/hpc/aggregate_eval_shards.py")
evaluate = load_script("lrvla_evaluate_resume", "scripts/evaluate.py")


def workflow_fixture(tmp_path):
    run = tmp_path / "benchmark"
    run.mkdir()
    shards = [{"tasks": members, "output": str(run / "shards" / f"shard-{index:03d}")}
              for index, members in enumerate(submit.split_tasks(list(TASKS), 5), 1)]
    return {"schema": "lrvla-pbs-evaluation-v1", "run_dir": str(run), "checkpoint": "/fixed/old1k/hf_ckpt",
            "precision": "fp32", "inference_backend": "native", "trials": 100, "settings": list(INSTRUCTIONS),
            "use_length": 50, "tasks": list(TASKS), "shards": shards}


def add_plan(workflow, shard, *, identity="same-model"):
    run = Path(shard["output"])
    run.mkdir(parents=True, exist_ok=True)
    plan = {key: workflow[key] for key in ("checkpoint", "precision", "trials", "settings", "inference_backend", "use_length")}
    plan.update(tasks=shard["tasks"], checkpoint_identity={"sha256": identity, "shards": []})
    (run / "evaluation_plan.json").write_text(json.dumps(plan))
    return plan


def add_counts(run, setting, task, success, trials, code):
    logfile = Path(run) / setting / "eval_logs" / f"{task}.log"
    logfile.parent.mkdir(parents=True, exist_ok=True)
    logfile.write_text(f"Success rate: {success}/{trials} => observed\n")
    logfile.with_suffix(".status.json").write_text(json.dumps({"exit_code": code}))


@pytest.mark.parametrize("size", [1, 5, 7, 50])
def test_task_partition_covers_all_ten_thousand_scenes_without_repeats(size):
    shards = submit.split_tasks(list(TASKS), size)
    recovered = [task for shard in shards for task in shard]
    assert recovered == list(TASKS)
    assert len(recovered) == len(set(recovered)) == 50
    scenes = [(task, setting, scene) for shard in shards for task in shard
              for setting in INSTRUCTIONS for scene in range(100)]
    assert len(scenes) == len(set(scenes)) == 10000


def test_partial_shard_reports_observed_counts_without_a_benchmark_score(tmp_path):
    workflow = workflow_fixture(tmp_path)
    first = workflow["shards"][0]
    add_plan(workflow, first)
    add_counts(first["output"], "demo_clean", first["tasks"][0], 12, 100, 0)
    add_counts(first["output"], "demo_randomized", first["tasks"][0], 3, 30, 1)
    report = aggregate.collect_workflow(workflow)
    assert report["settings"]["demo_clean"]["trials"] == 100
    assert report["settings"]["demo_randomized"]["trials"] == 30
    assert sum(setting["expected_trials"] for setting in report["settings"].values()) == 10000
    assert not report["complete"] and not report["comparable_to_release"]
    assert all(setting["benchmark_success_rate"] is None for setting in report["settings"].values())


def test_only_a_complete_consistent_native_protocol_reports_the_full_score(tmp_path):
    workflow = workflow_fixture(tmp_path)
    for shard in workflow["shards"]:
        add_plan(workflow, shard)
        for setting in INSTRUCTIONS:
            for task in shard["tasks"]:
                add_counts(shard["output"], setting, task, 10, 100, 0)
    report = aggregate.collect_workflow(workflow)
    assert report["complete"] and report["full_50_task_benchmark"] and report["comparable_to_release"]
    assert all(setting["trials"] == 5000 and setting["benchmark_success_rate"] == 0.1
               for setting in report["settings"].values())


@pytest.mark.parametrize("use_length,comparable", [(10, False), (50, True)])
def test_single_and_sharded_reports_agree_on_action_execution_length(tmp_path, use_length, comparable):
    workflow = workflow_fixture(tmp_path)
    workflow["use_length"] = use_length
    for shard in workflow["shards"]:
        add_plan(workflow, shard)
        for setting in INSTRUCTIONS:
            for task in shard["tasks"]:
                add_counts(shard["output"], setting, task, 10, 100, 0)
                add_counts(tmp_path, setting, task, 10, 100, 0)
    direct = aggregate_results(tmp_path, list(TASKS), list(INSTRUCTIONS), "fp32", use_length=use_length)
    sharded = aggregate.collect_workflow(workflow)
    assert direct["complete"] and sharded["complete"]
    assert direct["use_length"] == sharded["use_length"] == use_length
    assert direct["comparable_to_release"] is sharded["comparable_to_release"] is comparable


@pytest.mark.parametrize("change", ["duplicate", "missing"])
def test_invalid_shard_coverage_cannot_claim_completion(tmp_path, change):
    workflow = workflow_fixture(tmp_path)
    if change == "duplicate":
        workflow["shards"].append(deepcopy(workflow["shards"][0]))
    else:
        workflow["shards"].pop()
    with pytest.raises(ValueError, match="repeats task|coverage differs"):
        aggregate.collect_workflow(workflow)


@pytest.mark.parametrize("key,value", [("precision", "bf16"), ("inference_backend", "streamed"),
                                        ("checkpoint", "/different/model"), ("trials", 2), ("use_length", 25)])
def test_aggregation_rejects_mixed_protocols(tmp_path, key, value):
    workflow = workflow_fixture(tmp_path)
    shard = workflow["shards"][0]
    plan = add_plan(workflow, shard)
    plan[key] = value
    (Path(shard["output"]) / "evaluation_plan.json").write_text(json.dumps(plan))
    with pytest.raises(ValueError, match=key):
        aggregate.collect_workflow(workflow)


def test_identical_paths_do_not_hide_different_model_bytes(tmp_path):
    workflow = workflow_fixture(tmp_path)
    add_plan(workflow, workflow["shards"][0], identity="weights-A")
    add_plan(workflow, workflow["shards"][1], identity="weights-B")
    with pytest.raises(ValueError, match="different checkpoint bytes"):
        aggregate.collect_workflow(workflow)


@pytest.mark.parametrize("trials,code,complete", [(100, 0, True), (99, 0, False), (100, 1, False), (2, 0, False)])
def test_only_full_successful_client_runs_can_be_skipped(tmp_path, trials, code, complete):
    add_counts(tmp_path, "demo_clean", "adjust_bottle", 1, trials, code)
    assert evaluation_task_complete(tmp_path, "demo_clean", "adjust_bottle", 100) is complete


def test_failed_logs_are_archived_without_destroying_completed_logs(tmp_path):
    add_counts(tmp_path, "demo_clean", "adjust_bottle", 12, 100, 0)
    add_counts(tmp_path, "demo_randomized", "adjust_bottle", 1, 12, 1)
    failed = tmp_path / "demo_randomized/eval_logs/adjust_bottle.log"
    evaluate.archive_previous_attempt(tmp_path, [failed, failed.with_suffix(".status.json")])
    assert evaluation_task_complete(tmp_path, "demo_clean", "adjust_bottle", 100)
    assert not failed.exists()
    archived = list((tmp_path / "previous_attempts").glob("*/demo_randomized/eval_logs/adjust_bottle.log"))
    assert len(archived) == 1 and "1/12" in archived[0].read_text()


@pytest.mark.parametrize("key,value", [("checkpoint", "/another/model"), ("precision", "bf16"),
                                        ("inference_backend", "streamed"), ("trials", 2), ("use_length", 25)])
def test_resume_refuses_to_reuse_completed_scenes_from_another_protocol(tmp_path, key, value):
    workflow = workflow_fixture(tmp_path)
    shard = workflow["shards"][0]
    previous = add_plan(workflow, shard)
    requested = {**previous, key: value}
    with pytest.raises(ValueError, match=key):
        evaluate.prepare_resumed_plan(Path(shard["output"]), requested)


def test_generated_jobs_preserve_pbs_gpu_binding_and_only_route_valid_long_jobs(tmp_path, monkeypatch):
    args = ["--project", "personal", "--workdir", str(ROOT), "--conda-root", "/home/user/miniconda3",
            "--checkpoint", "/fixed/old1k/hf_ckpt", "--run-dir", str(tmp_path / "run"),
            "--output-dir", str(tmp_path / "pbs"), "--dependency", "afterok:25806020.pbs101"]
    launched = []
    monkeypatch.setattr(submit.subprocess, "run", lambda *args, **kwargs: launched.append(args))
    assert submit.main(args) == 0
    assert not launched
    scripts = list((tmp_path / "pbs").glob("lrvlaeval[0-9]*.pbs"))
    assert len(scripts) == 10
    for script in scripts:
        text = script.read_text()
        assert "#PBS -q normal" in text and "walltime=24:00:00" in text
        assert "select=1:ncpus=16:mem=110gb:ngpus=1" in text
        assert '--gpu "$CUDA_VISIBLE_DEVICES"' in text
        assert "CUDA_VISIBLE_DEVICES=0" not in text
        assert 'LRVLA_EVAL_PORT=$((20000 + ${PBS_JOBID%%.*} % 30000))' in text
        assert '--port "$LRVLA_EVAL_PORT"' in text
        assert "--native-server" in text and "--precision fp32" in text and "--trials 100" in text
        assert "--use-length 50" in text and "--skip-completed" in text
    with pytest.raises(SystemExit):
        submit.main([*args, "--walltime", "02:00:00"])


def test_finished_shard_does_not_start_a_renderer_or_load_the_model_again(tmp_path, monkeypatch):
    model_run = tmp_path / "model"
    checkpoint = model_run / "checkpoints/global_step_1000/hf_ckpt"
    checkpoint.mkdir(parents=True)
    run = tmp_path / "finished"
    run.mkdir()
    identity = {"sha256": "fixed-model", "shards": []}
    plan = {"checkpoint": str(checkpoint), "precision": "fp32", "trials": 100,
            "tasks": ["adjust_bottle"], "settings": list(INSTRUCTIONS), "inference_backend": "native",
            "use_length": 50, "checkpoint_identity": identity}
    (run / "evaluation_plan.json").write_text(json.dumps(plan))
    for setting in INSTRUCTIONS:
        add_counts(run, setting, "adjust_bottle", 2, 100, 0)
    monkeypatch.setattr(sys, "argv", ["evaluate.py", "--checkpoint", str(checkpoint), "--output", str(run),
                                     "--tasks", "adjust_bottle", "--native-server", "--skip-completed"])
    monkeypatch.setattr(evaluate, "linux_executable", lambda _name, preferred: preferred)
    monkeypatch.setattr(evaluate, "base_checkpoint_identity", lambda _path: identity)
    monkeypatch.setattr(evaluate.subprocess, "Popen", lambda *_args, **_kwargs: pytest.fail("Completed shard relaunched a process"))
    evaluate.main()
    assert json.loads((run / "results.json").read_text())["complete"]


def test_workflow_and_evaluator_agree_when_runs_are_on_a_scratch_symlink(tmp_path):
    scratch = tmp_path / "scratch"
    scratch.mkdir()
    home_runs = tmp_path / "runs"
    home_runs.symlink_to(scratch, target_is_directory=True)
    checkpoint = home_runs / "old1k/checkpoints/global_step_1000/hf_ckpt"
    checkpoint.mkdir(parents=True)
    args = ["--project", "personal", "--workdir", str(ROOT), "--conda-root", "/home/user/miniconda3",
            "--checkpoint", str(checkpoint), "--run-dir", str(home_runs / "benchmark"),
            "--output-dir", str(tmp_path / "pbs")]
    assert submit.main(args) == 0
    workflow = json.loads((scratch / "benchmark/evaluation_workflow.json").read_text())
    assert workflow["checkpoint"] == str(checkpoint.resolve())
    assert workflow["run_dir"] == str(scratch / "benchmark")
    assert all(shard["output"].startswith(str(scratch / "benchmark")) for shard in workflow["shards"])


def test_preparing_two_models_does_not_overwrite_the_other_models_job_scripts(tmp_path):
    workflows = []
    for model in ("old1k", "new10k"):
        run = tmp_path / model
        assert submit.main(["--project", "personal", "--workdir", str(ROOT),
                            "--conda-root", "/home/user/miniconda3", "--checkpoint", f"/fixed/{model}/hf_ckpt",
                            "--run-dir", str(run)]) == 0
        workflows.append(json.loads((run / "evaluation_workflow.json").read_text()))
    first_scripts = {shard["script"] for shard in workflows[0]["shards"]}
    second_scripts = {shard["script"] for shard in workflows[1]["shards"]}
    assert first_scripts.isdisjoint(second_scripts)
    for workflow in workflows:
        for shard in workflow["shards"]:
            assert f'--checkpoint {workflow["checkpoint"]}' in Path(shard["script"]).read_text()
        assert workflow["run_dir"] in Path(workflow["aggregate_script"]).read_text()
