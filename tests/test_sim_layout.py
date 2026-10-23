"""Policy-server and simulator-only GPU roles for FP32 evaluation on 40 GB GPUs."""
import json
import os
from pathlib import Path
import signal
import sys
from types import SimpleNamespace

import pytest
import yaml

from lrvla import parallel_evaluation as parallel
from lrvla.evaluation_workflow import prepare_workflow
from lrvla.simulation_workers import run_clients

DEVICES = [{"uuid": f"GPU-{name}", "pci": f"0000:0{index}:00.0"} for index, name in enumerate("abcd", 1)]
LAYOUT = {"server_gpus": 2, "sims_on_server_gpu": 1, "sims_per_sim_gpu": 3}


def test_split_roles_gives_each_server_its_own_gpu_and_one_simulator_gpu():
    servers, slots = parallel.split_roles(DEVICES, LAYOUT)
    assert [gpu["uuid"] for gpu in servers] == ["GPU-a", "GPU-b"]
    assert [gpu["uuid"] for gpu in slots["GPU-a"]] == ["GPU-a", "GPU-c", "GPU-c", "GPU-c"]
    assert [gpu["uuid"] for gpu in slots["GPU-b"]] == ["GPU-b", "GPU-d", "GPU-d", "GPU-d"]
    for layout in ({**LAYOUT, "server_gpus": 3}, {**LAYOUT, "server_gpus": 4}, {**LAYOUT, "sims_per_sim_gpu": 0},
                   {**LAYOUT, "sims_on_server_gpu": -1}, {**LAYOUT, "server_gpus": "2"}, None):
        with pytest.raises(ValueError):
            parallel.split_roles(DEVICES, layout)


def test_evaluator_arguments_carry_one_gpu_per_simulator_slot(tmp_path):
    plan = {"checkpoint": "/fixed/model", "precision": "fp32", "trials": 100, "inference_backend": "native",
            "settings": ["demo_clean", "demo_randomized"], "use_length": 50, "sim_workers": 4, "policy_rng": "scene"}
    _, slots = parallel.split_roles(DEVICES, LAYOUT)
    args = parallel.evaluation_arguments(plan, {"tasks": ["adjust_bottle"], "output": str(tmp_path)},
                                         DEVICES[0], 12345, slots["GPU-a"])
    assert args[args.index("--gpu") + 1] == "GPU-a"
    assert args[args.index("--sim-workers") + 1] == "4"
    assert args[args.index("--sim-gpus") + 1] == "GPU-a,GPU-c,GPU-c,GPU-c"
    legacy = parallel.evaluation_arguments(plan, {"tasks": ["adjust_bottle"], "output": str(tmp_path)}, DEVICES[0], 1)
    assert "--sim-gpus" not in legacy and legacy[legacy.index("--sim-workers") + 1] == "4"


def test_parallel_evaluator_sees_its_policy_and_simulator_gpus(tmp_path):
    evaluator = tmp_path / "fake_evaluate.py"
    evaluator.write_text('''import argparse,json,os
from pathlib import Path
EXPERT_SEED_POLICY = "test_policy"
RENDER_BINDING = True
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--sim-gpus")
    args,_=parser.parse_known_args()
    Path(args.output,"worker.json").write_text(json.dumps({"visible":os.environ["CUDA_VISIBLE_DEVICES"],
        "pci":os.environ["LRVLA_RENDER_PCI"],"sim_gpus":args.sim_gpus}))
''')
    plan = {"tasks": ["adjust_bottle", "click_bell"], "checkpoint": "/fixed/model", "precision": "fp32", "trials": 100,
            "settings": ["demo_clean"], "inference_backend": "native", "use_length": 50, "sim_workers": 4,
            "policy_rng": "scene", "expert_seed_policy": "test_policy",
            "shards": [{"tasks": [task], "output": str(tmp_path / task)} for task in ("adjust_bottle", "click_bell")]}
    servers, slots = parallel.split_roles(DEVICES, LAYOUT)
    assert parallel.run_parallel(plan, plan["shards"], servers, evaluate_script=evaluator,
                                 log_dir=tmp_path / "logs", sim_slots=slots) == 0
    rows = {json.loads((Path(unit["output"]) / "worker.json").read_text())["visible"] for unit in plan["shards"]}
    assert rows == {"GPU-a,GPU-c", "GPU-b,GPU-d"}


def test_worker_logs_use_short_shard_names_for_large_task_groups(tmp_path):
    evaluator = tmp_path / "fake_evaluate.py"
    evaluator.write_text('EXPERT_SEED_POLICY = "test_policy"\nRENDER_BINDING = True\ndef main():\n    print("ran")\n')
    tasks = [f"a_rather_long_robotwin_task_name_{index:02d}" for index in range(25)]
    plan = {"tasks": tasks, "checkpoint": "/fixed/model", "precision": "fp32", "trials": 10,
            "settings": ["demo_clean"], "inference_backend": "native", "use_length": 50,
            "expert_seed_policy": "test_policy", "shards": [{"tasks": tasks, "output": str(tmp_path / "group-001")}]}
    assert len("-".join(tasks)) > 255
    assert parallel.run_parallel(plan, plan["shards"], DEVICES[:1], evaluate_script=evaluator,
                                 log_dir=tmp_path / "logs") == 0
    assert "ran" in (tmp_path / "logs" / "group-001.log").read_text()


def test_clients_hold_one_slot_environment_per_process(tmp_path):
    commands = []
    for task in ("adjust_bottle", "click_bell", "lift_pot", "turn_switch", "stamp_seal"):
        record = tmp_path / f"{task}.json"
        code = ("import json,os,time; from pathlib import Path; start=time.time(); time.sleep(.3); "
                f"Path({str(record)!r}).write_text(json.dumps({{'gpu':os.environ['CUDA_VISIBLE_DEVICES'],"
                "'start':start,'end':time.time()}))")
        commands.append({"task": task, "setting": "demo_clean", "command": [sys.executable, "-c", code]})
    environments = [dict(os.environ, CUDA_VISIBLE_DEVICES=gpu) for gpu in ("GPU-a", "GPU-c", "GPU-c")]

    def stop(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)

    run_clients(commands, run=tmp_path, cwd=tmp_path, environment=dict(os.environ), environments=environments,
                workers=3, server=SimpleNamespace(poll=lambda: None), archive=lambda *args: None, stop=stop)
    rows = [json.loads((tmp_path / f"{c['task']}.json").read_text()) for c in commands]
    assert {row["gpu"] for row in rows} <= {"GPU-a", "GPU-c"}
    for gpu, capacity in (("GPU-a", 1), ("GPU-c", 2)):
        intervals = [(row["start"], row["end"]) for row in rows if row["gpu"] == gpu]
        for start, _ in intervals:
            assert sum(other_start <= start < other_end for other_start, other_end in intervals) <= capacity
    statuses = [json.loads((tmp_path / "demo_clean/eval_logs" / f"{c['task']}.status.json").read_text()) for c in commands]
    assert all(status["exit_code"] == 0 and status["gpu"] in {"GPU-a", "GPU-c"} for status in statuses)
    with pytest.raises(ValueError, match="one simulator environment per worker"):
        run_clients(commands, run=tmp_path, cwd=tmp_path, environment={}, environments=environments[:2], workers=3,
                    server=SimpleNamespace(poll=lambda: None), archive=lambda *args: None, stop=stop)


def test_workflow_records_validated_layout(tmp_path):
    base = {"tasks": ["adjust_bottle", "click_bell", "lift_pot", "turn_switch"], "settings": ["demo_clean"],
            "trials": 5, "backend": "native", "precision": "fp32", "use_length": 50, "num_steps": 10, "compile": False,
            "action_smoothing": "none", "sim_workers": 4, "policy_rng": "scene", "expert_cache_mode": "off"}
    config = tmp_path / "layout.yaml"
    config.write_text(yaml.safe_dump({**base, "sim_layout": LAYOUT}))
    workflow = json.loads(prepare_workflow(config, tmp_path / "model", tmp_path / "run", 2, root=tmp_path).read_text())
    assert workflow["sim_layout"] == LAYOUT and len(workflow["shards"]) == 2
    config.write_text(yaml.safe_dump({**base, "sim_layout": {"server_gpus": 2}}))
    with pytest.raises(ValueError, match="sim_layout"):
        prepare_workflow(config, tmp_path / "model", tmp_path / "other", 2, root=tmp_path)
    config.write_text(yaml.safe_dump(base))
    assert "sim_layout" not in json.loads(prepare_workflow(config, tmp_path / "model", tmp_path / "plain", 2,
                                                           root=tmp_path).read_text())
