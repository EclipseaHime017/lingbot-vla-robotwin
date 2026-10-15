"""Shared-service random isolation, real CPU client overlap and cached replay."""
import ast
import json
import os
from pathlib import Path
import random
import signal
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from lrvla.evaluation import adapt_eval_client
from lrvla.evaluation_runtime import add_runtime_features
from lrvla.evaluation_workflow import aggregate_workflow, prepare_workflow
from lrvla.policy_sessions import PolicySession
from lrvla.scene_bank import SceneBank, source_identity
from lrvla.simulation_workers import run_clients

ROOT = Path(__file__).resolve().parents[1]


def test_shared_policy_interleaving_preserves_each_scene_noise_and_global_rng(monkeypatch):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    class Policy:
        chunk_ret = True
        def infer(self, observation):
            return {"action": None if observation.get("reset") else torch.randn(14)}
    policy = Policy()
    a, b = PolicySession(policy), PolicySession(policy)
    original = torch.get_rng_state()
    a.infer({"reset": True, "robo_name": "robotwin", "rng_seed": 5})
    b.infer({"reset": True, "robo_name": "robotwin", "rng_seed": 9})
    interleaved = [a.infer({})["action"], b.infer({})["action"], a.infer({})["action"], b.infer({})["action"]]
    assert torch.equal(torch.get_rng_state(), original)
    for session, seed, positions in ((a, 5, [0, 2]), (b, 9, [1, 3])):
        session.infer({"reset": True, "robo_name": "robotwin", "rng_seed": seed})
        for position in positions:
            assert torch.equal(session.infer({})["action"], interleaved[position])
    with pytest.raises(ValueError, match="checkpoint"):
        a.infer({"reset": True, "path_to_pi_model": "/other"})
    with pytest.raises(ValueError, match="robotwin"):
        a.infer({"reset": True, "robo_name": "other", "rng_seed": 5})
    with pytest.raises(ValueError, match="Reset"):
        PolicySession(policy).infer({})
    with pytest.raises(ValueError, match="chunk_ret"):
        PolicySession(SimpleNamespace(chunk_ret=False))


@pytest.mark.parametrize("server_dies", [False, True])
def test_real_simulator_processes_overlap_and_are_cleaned_on_server_failure(tmp_path, server_dies):
    commands = []
    for task in ("adjust_bottle", "click_bell"):
        pidfile = tmp_path / f"{task}.json"
        code = ("import json,os,time; from pathlib import Path; "
                f"Path({str(pidfile)!r}).write_text(json.dumps({{'pid':os.getpid(),'start':time.time()}})); "
                f"time.sleep({60 if server_dies else .25})")
        commands.append({"task": task, "setting": "demo_clean", "command": [sys.executable, "-c", code]})
    def stop(process):
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            process.wait(timeout=5)
    # Let both clients start before simulating the shared server's failure.
    server = SimpleNamespace(poll=lambda: 1 if server_dies and all((tmp_path / f"{c['task']}.json").exists() for c in commands) else None)
    kwargs = dict(run=tmp_path, cwd=tmp_path, environment=dict(os.environ), workers=2, server=server,
                  archive=lambda *args: None, stop=stop)
    if server_dies:
        with pytest.raises(RuntimeError, match="server exited"):
            run_clients(commands, **kwargs)
    else:
        run_clients(commands, **kwargs)
    rows = [json.loads((tmp_path / f"{c['task']}.json").read_text()) for c in commands]
    assert len({row["pid"] for row in rows}) == 2
    if not server_dies:
        assert abs(rows[0]["start"] - rows[1]["start"]) < .2
    for command, row in zip(commands, rows):
        status = json.loads((tmp_path / "demo_clean/eval_logs" / f"{command['task']}.status.json").read_text())
        assert status["exit_code"] == (-signal.SIGTERM if server_dies else 0)
        with pytest.raises(ProcessLookupError):
            os.kill(row["pid"], 0)


def test_proxy_yaml_balancing_and_incomplete_aggregation(tmp_path):
    path = prepare_workflow(ROOT / "configs/eval_proxy.yaml", "/fixed/model", tmp_path / "proxy", 4)
    workflow = json.loads(path.read_text())
    assert len(workflow["tasks"]) * len(workflow["settings"]) * workflow["trials"] == 480
    assert len(workflow["shards"]) == 4 and workflow["sim_workers"] == 4 and workflow["policy_rng"] == "scene"
    assert sorted(t for s in workflow["shards"] for t in s["tasks"]) == sorted(workflow["tasks"])
    # Missing workers never become a fabricated score or a completed proxy.
    report = aggregate_workflow(workflow)
    assert not report["complete"] and not report["comparable_to_release"]
    assert report["settings"]["demo_clean"]["expected_trials"] == 240
    workflow["submission_disabled"] = True
    path.write_text(json.dumps(workflow))
    with pytest.raises(ValueError, match="stopped"):
        prepare_workflow(ROOT / "configs/eval_proxy.yaml", "/fixed/model", tmp_path / "proxy", 4)


def test_cache_identity_tracks_robot_physics_configs(tmp_path):
    config = tmp_path / "assets/embodiments/robot/config.yml"
    config.parent.mkdir(parents=True)
    config.write_text("friction: 1\n")
    first = source_identity(tmp_path)
    config.write_text("friction: 2\n")
    assert source_identity(tmp_path) != first


def test_proxy_complete_480_results_require_matching_identity_and_protocol(tmp_path):
    workflow = json.loads(prepare_workflow(ROOT / "configs/eval_proxy.yaml", "/fixed/model", tmp_path, 4).read_text())
    for shard in workflow["shards"]:
        directory = Path(shard["output"])
        directory.mkdir()
        plan = {**workflow, "tasks": shard["tasks"], "checkpoint_identity": {"sha256": "fixed-bytes"}}
        (directory / "evaluation_plan.json").write_text(json.dumps(plan))
        for setting in workflow["settings"]:
            logs = directory / setting / "eval_logs"
            logs.mkdir(parents=True)
            for task in shard["tasks"]:
                (logs / f"{task}.log").write_text("Success rate: 10/20 => 50%\n")
                (logs / f"{task}.status.json").write_text('{"exit_code": 0}')
    report = aggregate_workflow(workflow)
    assert report["complete"] and not report["full_50_task_benchmark"] and not report["comparable_to_release"]
    assert sum(v["trials"] for v in report["settings"].values()) == 480
    path = Path(workflow["shards"][0]["output"]) / "evaluation_plan.json"
    original = json.loads(path.read_text())
    path.write_text(json.dumps({**original, "num_steps": 20}))
    with pytest.raises(ValueError, match="num_steps"):
        aggregate_workflow(workflow)
    path.write_text(json.dumps({**original, "checkpoint_identity": {"sha256": "other-bytes"}}))
    with pytest.raises(ValueError, match="checkpoint bytes"):
        aggregate_workflow(workflow)


def test_generated_cached_loop_replays_seed_instruction_and_initial_rgb_without_expert(tmp_path, monkeypatch, capsys):
    source = adapt_eval_client((ROOT / "vendor/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py").read_text(), 2)
    source = add_runtime_features(source, scene_rng=True, expert_cache=True, timing=True)
    function = next(n for n in ast.parse(source).body if isinstance(n, ast.FunctionDef) and n.name == "eval_policy")
    import lrvla.episode_metrics as metrics
    import lrvla.policy_sessions as sessions
    import lrvla.scene_bank as bank_module
    monkeypatch.setitem(sys.modules, "episode_metrics", metrics)
    monkeypatch.setitem(sys.modules, "policy_sessions", sessions)
    monkeypatch.setitem(sys.modules, "scene_bank", bank_module)
    monkeypatch.setenv("LRVLA_EXPERT_CACHE", str(tmp_path))
    monkeypatch.setenv("LRVLA_EXPERT_CACHE_IDENTITY", json.dumps({"sources": "test-pinned"}))
    class Environment:
        def __init__(self):
            self.expert_starts = 0
            self.render_freq = 0
            self.eval_video_path = None
            self.instructions, self.observations = [], []
            self.info = {"info": {}}
        def setup_demo(self, **kwargs):
            self.seed = kwargs["seed"]
            np.random.seed(self.seed)
            np.random.rand(3)  # Scene generation consumes RNG before instructions.
            self.take_action_cnt, self.step_lim = 0, 1
        def play_once(self):
            self.expert_starts += 1
            self.plan_success = self.seed != 100000
            return {"info": {"{A}": "object/001", "{a}": "left"}}
        def close_env(self, **kwargs): pass
        def set_instruction(self, instruction):
            self.instructions.append(str(instruction))
        def get_instruction(self): return self.instructions[-1]
        def get_obs(self):
            observation = {"joint_action": {"vector": np.random.rand(14)}, "observation": {
                key: {"rgb": np.random.randint(0, 255, (2, 2, 3), dtype=np.uint8)}
                for key in ("head_camera", "left_camera", "right_camera")}}
            self.observations.append(bank_module.observation_identity(observation))
            return observation
        def take_action(self, action): self.take_action_cnt += 1
        def check_success(self): return True
    class Policy:
        def infer(self, observation): return {"action": np.ones(14), "server_timing": {"infer_ms": 2}}
    def generate(task, episode_info, limit):
        assert limit == 2 and episode_info == [{"{A}": "object/001", "{a}": "left"}]
        descriptions = ["move left", "grasp left", "pick left"]
        random.shuffle(descriptions)
        return [{"seen": descriptions[:limit]}]
    scope = {"np": np, "os": os, "Path": Path, "UnStableError": type("UnStableError", (Exception,), {}),
             "generate_episode_descriptions": generate}
    exec(compile(ast.Module(body=[function], type_ignores=[]), "cached_client", "exec"), scope)
    args = {"task_name": "adjust_bottle", "policy_name": "test", "clear_cache_freq": 5,
            "render_freq": 0, "task_config": "demo_clean", "ckpt_setting": "fixed"}
    def run(mode):
        monkeypatch.setenv("LRVLA_EXPERT_CACHE_MODE", mode)
        env = Environment()
        scope["eval_policy"]("adjust_bottle", env, args.copy(), Policy(), 100000, test_num=2,
                             instruction_type="seen", usr_args={"robo_name": "robotwin"})
        return env
    random.seed(42)
    recorded = run("record")
    random.seed(999)  # The caller/worker's Python state must not affect replay.
    replayed = run("replay")
    assert recorded.expert_starts == 3 and replayed.expert_starts == 0
    assert replayed.instructions == recorded.instructions and replayed.observations == recorded.observations
    assert replayed.test_num == recorded.test_num == 2
    assert capsys.readouterr().out.count("LRVLA_EPISODE_METRICS=") == 4
    data = json.loads(next(tmp_path.glob("*/scenes.json")).read_text())
    assert [r["policy_seed"] for r in data["scenes"]] == [100001, 100002]
    assert [len(r["attempts"]) for r in data["scenes"]] == [2, 1]


def test_partial_or_invalid_scene_bank_cannot_replay(tmp_path):
    bank = SceneBank(tmp_path, {"sources": "fixed"}, "record", task="adjust_bottle", setting="demo_clean", trials=1, start_seed=100000)
    row = bank.expert_done(0, 100000, 100000, {"info": {}}, [{"seed": 100000, "success": False}])
    row["ready"] = True
    bank.save()
    bank.close()
    with pytest.raises(ValueError, match="seed chain"):
        SceneBank(tmp_path, {"sources": "fixed"}, "replay", task="adjust_bottle", setting="demo_clean", trials=1, start_seed=100000)
    # A failed constructor must release its shared lock too.
    with pytest.raises(ValueError, match="overwrite"):
        SceneBank(tmp_path, {"sources": "fixed"}, "record", task="adjust_bottle", setting="demo_clean", trials=1, start_seed=100000)
