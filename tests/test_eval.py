import ast
import json
import importlib.util
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.evaluation import adapt_eval_client, aggregate_results, export_official_results, parse_success_counts
from lrvla.data import TASKS
from lrvla.scene_metadata import object_expressions, recover_scene_info

spec = importlib.util.spec_from_file_location("lrvla_evaluate_cli", ROOT / "scripts/evaluate.py")
evaluate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluate)


class EvaluationTests(unittest.TestCase):
    def test_conda_root_override_and_home_default(self):
        with mock.patch.dict(os.environ, {"CONDA_ROOT": "~/portable-miniconda"}):
            self.assertEqual(evaluate.default_conda(), Path.home() / "portable-miniconda/bin/conda")
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertEqual(evaluate.default_conda(), Path.home() / "miniconda3/bin/conda")
        with mock.patch.dict(os.environ, {"CONDA_ROOT": ""}):
            self.assertEqual(evaluate.default_conda(), Path.home() / "miniconda3/bin/conda")

    def test_checkpoint_attestation_is_after_rendering_and_before_server(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_run = root / "model"
            checkpoint = model_run / "checkpoints/global_step_1/hf_ckpt"
            checkpoint.mkdir(parents=True)
            (checkpoint / "model.safetensors").write_bytes(b"test fixture")
            (model_run / "lingbotvla_cli.yaml").write_text("{}")
            configs = model_run / "configs/robot_configs"
            configs.mkdir(parents=True)
            (configs / "robotwin.yaml").write_text("{}")
            run = root / "evaluation"
            arguments = ["evaluate.py", "--checkpoint", str(checkpoint), "--tasks", "adjust_bottle",
                         "--setting", "demo_clean", "--output", str(run)]
            probe = mock.Mock(return_value={"ok": False, "status": "failed"})
            doctor = SimpleNamespace(isolated_probe=probe, RENDER_CODE="test renderer")
            identity = {"sha256": "test-identity", "shards": []}
            with mock.patch.object(sys, "argv", arguments), mock.patch.dict(sys.modules, {"doctor": doctor}), \
                 mock.patch.dict(os.environ, {"CONDA_ROOT": str(root / "moved-miniconda"),
                                              "PATH": "/usr/bin:/mnt/c/Windows/System32",
                                              "LD_LIBRARY_PATH": "/mnt/c/Tools/lib:/usr/lib",
                                              "CC": "/mnt/c/Tools/cl.exe"}), \
                 mock.patch.object(evaluate, "linux_executable", side_effect=lambda name, preferred: preferred), \
                 mock.patch.object(evaluate, "upstream_revision", return_value=evaluate.ROBOTWIN_REVISION), \
                 mock.patch.object(evaluate, "base_checkpoint_identity", return_value=identity) as attest, \
                 mock.patch.object(evaluate.subprocess, "Popen") as launch, \
                 mock.patch.object(evaluate.socket, "socket"):
                with self.assertRaisesRegex(RuntimeError, "Model loading was not started"):
                    evaluate.main()
                attest.assert_not_called()
                launch.assert_not_called()
                plan = json.loads((run / "evaluation_plan.json").read_text())
                self.assertNotIn("checkpoint_identity", plan)
                self.assertEqual(plan["inference_command"][0], str(root / "moved-miniconda/envs/lingbot-vla/bin/python"))
                self.assertEqual(plan["evaluation_commands"][0]["command"][0], str(root / "moved-miniconda/envs/robotwin-sim/bin/python"))
                sim_environment = probe.call_args.kwargs["env"]
                self.assertEqual(sim_environment["PATH"], str(root / "moved-miniconda/envs/robotwin-sim/bin") + ":/usr/bin:/mnt/c/Windows/System32")
                self.assertEqual(sim_environment["LD_LIBRARY_PATH"], str(root / "moved-miniconda/envs/robotwin-sim/lib") + ":/mnt/c/Tools/lib:/usr/lib")
                self.assertEqual(sim_environment["CC"], "/mnt/c/Tools/cl.exe")
                probe.return_value = {"ok": True, "status": "passed"}
                def inspect_before_launch(*args, **kwargs):
                    self.assertEqual(json.loads((run / "evaluation_plan.json").read_text())["checkpoint_identity"], identity)
                    self.assertEqual(kwargs["env"]["PATH"], str(root / "moved-miniconda/envs/lingbot-vla/bin") + ":/usr/bin:/mnt/c/Windows/System32")
                    self.assertEqual(kwargs["env"]["LD_LIBRARY_PATH"], str(root / "moved-miniconda/envs/lingbot-vla/lib") + ":/mnt/c/Tools/lib:/usr/lib")
                    raise RuntimeError("server launch observed")
                launch.side_effect = inspect_before_launch
                with self.assertRaisesRegex(RuntimeError, "server launch observed"):
                    evaluate.main()
                attest.assert_called_once_with(checkpoint)

    def test_all_official_tasks_have_object_only_seen_unseen_templates(self):
        import re
        root = ROOT / "vendor/RoboTwin"
        for task in TASKS:
            keys = {key for key, _ in object_expressions(task, str(root))}
            instructions = json.loads((root / "description/task_instruction" / f"{task}.json").read_text())
            for pool in ("seen", "unseen"):
                self.assertTrue(any(set(re.findall(r"\{[^}]+\}", text)) == keys for text in instructions[pool]), f"{task}/{pool}")

    def test_counts_parse_last_ansi_line(self):
        self.assertEqual(parse_success_counts("Success rate: 1/3 => 33.3%\nSuccess rate: \x1b[96m9/100\x1b[0m => 9.0%"), (9, 100))

    def test_partial_run_never_claims_full_score(self):
        with tempfile.TemporaryDirectory() as root:
            run = Path(root)
            directory = run / "demo_clean/eval_logs"
            directory.mkdir(parents=True)
            (directory / "adjust_bottle.log").write_text("Success rate: 2/3 => 66.7%")
            (directory / "adjust_bottle.status.json").write_text('{"exit_code": 1}')
            report = aggregate_results(run, ["adjust_bottle"], ["demo_clean", "demo_randomized"], "bf16")
            self.assertFalse(report["complete"])
            self.assertFalse(report["comparable_to_release"])
            self.assertIsNone(report["settings"]["demo_clean"]["benchmark_success_rate"])
            payload = export_official_results(report, run / "submission.json", "my-team", ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json")
            self.assertEqual(payload["results"]["clean"]["adjust_bottle"], {"attempts": 3, "successes": 2})
            self.assertEqual(payload["results"]["randomized"]["adjust_bottle"], {"attempts": 0, "successes": 0})
            self.assertEqual(set(payload), {"schema_version", "team_id", "results"})
            self.assertEqual(len(payload["results"]["clean"]), 50)

    def test_expert_failure_retains_scene_after_five_attempts(self):
        source = adapt_eval_client((ROOT / "vendor/lingbot-vla-v2/experiment/robotwin/eval_policy_client_lingbotvla.py").read_text(), 1)
        tree = ast.parse(source)
        function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "eval_policy")
        # Execute the actual adapted official loop with a minimal simulator, so
        # the test proves policy testing survives exhausted expert collection.
        class Environment:
            def __init__(self):
                self.expert_starts = 0
                self.policy_actions = 0
                self.seeds = []
                self.info = {"info": {}}
                self.model_id = 13
                self.eval_video_path = None
                self.take_action_cnt = 0
                self.step_lim = 1
                self.render_freq = 0
            def setup_demo(self, **kwargs):
                self.seeds.append(kwargs["seed"])
            def play_once(self):
                self.expert_starts += 1
                raise RuntimeError("expert planning failed")
            def close_env(self, **kwargs):
                pass
            def set_instruction(self, instruction):
                self.instruction = instruction
            def get_instruction(self):
                return self.instruction
            def get_obs(self):
                return {"observation": {key: {"rgb": "image"} for key in ("head_camera", "left_camera", "right_camera")}, "joint_action": {"vector": "state"}}
            def take_action(self, action):
                self.policy_actions += 1
                self.take_action_cnt += 1
            def check_success(self):
                return self.policy_actions > 0
        class Action:
            shape = (14,)
        class Policy:
            def infer(self, observation):
                return {"action": Action(), "server_timing": 0}
        class NP:
            class random:
                choice = staticmethod(lambda x: x[0])
        def generate(task, episodes, *args):
            self.assertEqual(episodes, [{"{A}": "001_bottle/base13"}])
            return [{"seen": ["instruction"]}]
        # The runtime puts this pure stdlib helper next to the generated client.
        import lrvla.scene_metadata as helper
        sys.modules["scene_metadata"] = helper
        scope = {"np": NP(), "os": __import__("os"), "Path": Path,
                 "parent_directory": str(ROOT / "vendor/RoboTwin/script"), "generate_episode_descriptions": generate}
        exec(compile(ast.Module(body=[function], type_ignores=[]), "adapted_client", "exec"), scope)
        env = Environment()
        args = {"task_name": "adjust_bottle", "policy_name": "lingbot", "clear_cache_freq": 5, "render_freq": 0,
                "task_config": "demo_clean", "ckpt_setting": "cotrain"}
        scope["eval_policy"]("adjust_bottle", env, args, Policy(), 100000, test_num=1, instruction_type="seen", usr_args={"robo_name": "robotwin"})
        self.assertEqual(env.expert_starts, 5)
        self.assertEqual(env.policy_actions, 1)
        self.assertEqual(env.test_num, 1)
        self.assertEqual(env.suc, 1)
        self.assertEqual(set(env.seeds), {100000})


if __name__ == "__main__":
    unittest.main()
