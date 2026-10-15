"""Independent GPU assignments, legacy protocol reuse and real process isolation."""
import ast
import json
import os
from pathlib import Path
import subprocess
from types import ModuleType, SimpleNamespace
import sys

import pytest

from lrvla import parallel_evaluation as parallel
from lrvla.render_device import bind_render_device, normalized_pci


def workflow(tmp_path):
    tasks = ["adjust_bottle", "click_bell"]
    return {"tasks": tasks, "checkpoint": "/fixed/model", "precision": "fp32", "trials": 100,
            "settings": ["demo_clean", "demo_randomized"], "inference_backend": "native", "use_length": 50,
            "expert_seed_policy": "test_policy", "shards": [
                {"tasks": [task], "output": str(tmp_path / task)} for task in tasks]}


def test_pci_normalization_and_allocated_gpu_visibility(monkeypatch):
    assert normalized_pci("00000000:AB:00.0") == "0000:ab:00.0"
    with pytest.raises(ValueError, match="PCI"):
        normalized_pci("GPU-abc")
    monkeypatch.setattr(parallel, "linux_executable", lambda _: Path("/usr/bin/nvidia-smi"))
    def query(command, **kwargs):
        key = command[-1]
        mapping = {"0": "GPU-a, 00000000:01:00.0", "GPU-a": "GPU-a, 00000000:01:00.0",
                   "1": "GPU-b, 00000000:02:00.0", "GPU-b": "GPU-b, 00000000:02:00.0"}
        return SimpleNamespace(stdout=mapping[key])
    monkeypatch.setattr(parallel.subprocess, "run", query)
    devices = parallel.assigned_devices(environment={"CUDA_VISIBLE_DEVICES": "GPU-a,GPU-b"})
    assert [row["uuid"] for row in devices] == ["GPU-a", "GPU-b"]
    assert parallel.assigned_devices("1", environment={"CUDA_VISIBLE_DEVICES": "GPU-b"}) == [devices[1]]
    with pytest.raises(ValueError, match="outside"):
        parallel.assigned_devices("0", environment={"CUDA_VISIBLE_DEVICES": "GPU-b"})
    with pytest.raises(ValueError, match="Duplicate"):
        parallel.assigned_devices("0,GPU-a", environment={})


def test_parallel_selection_and_legacy_profile_remain_fixed(tmp_path):
    plan = workflow(tmp_path)
    assert parallel.select_units(plan, "all") == plan["shards"]
    args = parallel.evaluation_arguments(plan, plan["shards"][0], {"uuid": "GPU-a"}, 12345)
    assert args[args.index("--gpu") + 1] == "GPU-a"
    assert "--native-server" in args and "--num-steps" not in args
    modern = {**plan, "num_steps": 20, "compile": False, "action_smoothing": "none"}
    args = parallel.evaluation_arguments(modern, plan["shards"][0], {"uuid": "GPU-b"}, 12346)
    assert args[args.index("--num-steps") + 1] == "20" and "--no-compile" in args
    plan["shards"][1]["output"] = plan["shards"][0]["output"]
    with pytest.raises(ValueError, match="Overlapping"):
        parallel.select_units(plan, "all")
    plan["shards"] = [{"tasks": plan["tasks"], "output": str(tmp_path / "combined")}]
    with pytest.raises(ValueError, match="whole shards"):
        parallel.select_units(plan, "click_bell")


def test_binding_injection_preserves_future_imports():
    source = '"""Original client."""\nfrom __future__ import annotations\nimport sapien\n'
    result = parallel.inject_binding(source)
    tree = ast.parse(result)
    assert tree.body[1].module == "__future__"
    assert result.index("bind_render_device()") < result.rindex("import sapien")


def test_sapien_binding_selects_first_renderer_and_default_scene(monkeypatch):
    class Device:
        def __init__(self, alias):
            self.pci_string = alias.removeprefix("pci:")
            self.cuda_id, self.name = 0, "Assigned GPU"
        def can_render(self):
            return True
    class NativeRenderer:
        def __init__(self, device=None):
            self.selected_device = device
    class WrapperRenderer(NativeRenderer):
        pass
    class RenderSystem:
        def __init__(self, device=None):
            self.device = device
    class Physics:
        pass
    class Scene:
        def __init__(self, systems=None):
            self.systems = systems
            for system in systems or []:
                if isinstance(system, RenderSystem):
                    self.render_system = system
    sapien = ModuleType("sapien")
    sapien.Device = Device
    sapien.physx = SimpleNamespace(PhysxCpuSystem=Physics)
    sapien.render = SimpleNamespace(RenderSystem=RenderSystem)
    sapien.pysapien = SimpleNamespace(render=SimpleNamespace(SapienRenderer=NativeRenderer))
    renderer, scene = ModuleType("sapien.wrapper.renderer"), ModuleType("sapien.wrapper.scene")
    renderer.SapienRenderer, scene.Scene = WrapperRenderer, Scene
    monkeypatch.setitem(sys.modules, "sapien", sapien)
    monkeypatch.setitem(sys.modules, "sapien.wrapper.renderer", renderer)
    monkeypatch.setitem(sys.modules, "sapien.wrapper.scene", scene)
    bind_render_device("00000000:AB:00.0")
    assert WrapperRenderer().selected_device.pci_string == "0000:ab:00.0"
    assert Scene().render_system.device.pci_string == "0000:ab:00.0"
    physics = Scene([Physics()])
    assert not hasattr(physics, "render_system")
    bind_render_device("0000:ab:00.0")
    with pytest.raises(RuntimeError, match="switch"):
        bind_render_device("0000:cd:00.0")


def test_workers_are_distinct_processes_with_independent_gpu_port_and_output(tmp_path):
    evaluator = tmp_path / "fake_evaluate.py"
    evaluator.write_text('''import argparse,json,os,time
from pathlib import Path
EXPERT_SEED_POLICY = "test_policy"
def create_eval_runtime(*args, **kwargs):
    raise AssertionError("No simulator needed in this process-isolation test")
def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("--output")
    parser.add_argument("--port")
    args,_=parser.parse_known_args()
    started=time.time()
    time.sleep(.2)
    Path(args.output,"worker.json").write_text(json.dumps({"pid":os.getpid(),"gpu":os.environ["CUDA_VISIBLE_DEVICES"],"pci":os.environ["LRVLA_RENDER_PCI"],"port":args.port,"started":started,"finished":time.time()}))
''')
    plan = workflow(tmp_path)
    devices = [{"uuid": "GPU-a", "pci": "0000:01:00.0"}, {"uuid": "GPU-b", "pci": "0000:02:00.0"}]
    assert parallel.run_parallel(plan, plan["shards"], devices, evaluate_script=evaluator, log_dir=tmp_path / "logs") == 0
    results = [json.loads((Path(unit["output"]) / "worker.json").read_text()) for unit in plan["shards"]]
    assert len({row["pid"] for row in results}) == 2
    assert len({row["port"] for row in results}) == 2
    assert {row["gpu"] for row in results} == {"GPU-a", "GPU-b"}
    assert max(row["started"] for row in results) < min(row["finished"] for row in results)


def test_selected_evaluator_imports_its_own_protocol_release(tmp_path):
    root = tmp_path / "other_release"
    scripts, package = root / "scripts", root / "src/lrvla"
    scripts.mkdir(parents=True)
    package.mkdir(parents=True)
    (package / "evaluation.py").write_text('EXPERT_SEED_POLICY = "older_protocol"\n')
    (scripts / "doctor.py").write_text('RENDER_CODE = "pass\\n"\n')
    evaluator = scripts / "evaluate.py"
    evaluator.write_text('''import json
from pathlib import Path
import lrvla.evaluation as protocol
EXPERT_SEED_POLICY = protocol.EXPERT_SEED_POLICY
def create_eval_runtime(): pass
def main():
    print(json.dumps({"module": protocol.__file__, "policy": EXPERT_SEED_POLICY}))
''')
    env = dict(os.environ, PYTHONPATH=str(parallel.ROOT / "src"),
               LRVLA_EXPECT_EXPERT_SEED_POLICY="older_protocol")
    result = subprocess.run([sys.executable, str(parallel.ROOT / "scripts/evaluate_parallel.py"),
                             "--worker-entry", str(evaluator)], env=env, text=True, capture_output=True, check=True)
    assert json.loads(result.stdout) == {"module": str(package / "evaluation.py"), "policy": "older_protocol"}
