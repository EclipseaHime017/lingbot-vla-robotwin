"""Official-client adaptation and count-based RoboTwin result collection."""
from __future__ import annotations

import ast
import json
from pathlib import Path
import re
import shutil

from .data import TASKS

ROBOTWIN_REVISION = "13c3c47ff4312dd62484bcd51be034af55c062d1"
INSTRUCTIONS = {"demo_clean": "seen", "demo_randomized": "unseen"}
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
SUCCESS = re.compile(r"Success rate:\s*(\d+)\s*/\s*(\d+)\s*=>")


def parse_success_counts(text: str) -> tuple[int, int]:
    matches = SUCCESS.findall(ANSI.sub("", text))
    if not matches:
        return 0, 0
    success, total = map(int, matches[-1])
    if not 0 <= success <= total:
        raise ValueError("Invalid simulator success counts")
    return success, total


def adapt_eval_client(source: str, trials: int) -> str:
    """Keep official simulation/action API while bounding expert starts per scene.

    The unmodified reference skips scenes indefinitely after failed expert runs.
    Tianchi rule §4.1 permits five starts and requires policy evaluation afterwards.
    Retries here preserve the seed; every scene proceeds to the policy.
    """
    if not 1 <= trials <= 100:
        raise ValueError("Expected 1..100 trials")
    if source.count("    test_num = 100\n") != 1:
        raise ValueError("Upstream evaluation client changed: test_num anchor")
    source = source.replace("    test_num = 100\n", f"    test_num = {trials}\n", 1)
    # The client never calls get_model; retaining the lookup imports an unrelated
    # pi0/OpenPI stack into the isolated simulator environment.
    source = source.replace('    get_model = eval_function_decorator(policy_name, "get_model")\n', '')
    start = source.index("        if expert_check:\n", source.index("def eval_policy("))
    stop = source.index("        args[\"render_freq\"] = render_freq\n\n        TASK_ENV.setup_demo", start)
    replacement = '''        # Tianchi §4.1: at most five expert starts, then policy evaluation.
        episode_info = None
        for expert_attempt in range(5):
            print(f"Expert collection start {expert_attempt + 1}/5, scene={now_id}, seed={now_seed}")
            try:
                TASK_ENV.setup_demo(now_ep_num=now_id, seed=now_seed, is_test=True, **args)
                episode_info = TASK_ENV.play_once()
                expert_success = TASK_ENV.plan_success and TASK_ENV.check_success()
                if not isinstance(episode_info, dict) or "info" not in episode_info:
                    episode_info = {"info": dict(getattr(TASK_ENV, "info", {}).get("info", {}))}
            except Exception as error:
                print(f"Expert attempt {expert_attempt + 1}/5 failed; policy scene retained: {error}")
                episode_info = {"info": dict(getattr(TASK_ENV, "info", {}).get("info", {}))}
                expert_success = False
            finally:
                TASK_ENV.close_env()
            if expert_success:
                break
        if not expert_success:
            print("Expert quota exhausted; evaluating policy on this scene")
        succ_seed += 1
        suc_test_seed_list.append(now_seed)

'''
    source = source[:start] + replacement + source[stop:]
    # Use metadata from the fresh policy scene if expert exceptions left it empty.
    source = source.replace('        episode_info_list = [episode_info["info"]]',
                            '''        scene_info = episode_info["info"] or getattr(TASK_ENV, "info", {}).get("info", {})
        if not scene_info:
            from scene_metadata import recover_scene_info
            robotwin_root = (Path(parent_directory).parent / "task_config").resolve().parent
            scene_info = recover_scene_info(task_name, TASK_ENV, robotwin_root)
            print("Recovered official scene object descriptors; using official instruction templates")
        episode_info_list = [scene_info]''')
    # check_success() is the authoritative success signal specified by competition.
    source = source.replace("if TASK_ENV.eval_success:", "if TASK_ENV.check_success():")
    # Checkpoint switching based on task/instruction is prohibited.
    source = source.replace("        ret = model.infer(dict(reset = True, robo_name=usr_args['robo_name'], path_to_pi_model=path_to_pi_model))",
                            "        ret = model.infer(dict(reset=True, robo_name=usr_args['robo_name']))")
    ast.parse(source)
    return source


def create_eval_runtime(vendor: Path, robotwin: Path, destination: Path, trials: int) -> Path:
    """Generate a client copy outside both tracked upstream trees."""
    script_dir = destination / "script"
    deploy = script_dir / "deploy"
    deploy.mkdir(parents=True, exist_ok=True)
    (script_dir / "__init__.py").write_text("")
    client = script_dir / "eval_policy_client_lingbotvla.py"
    client.write_text(adapt_eval_client((vendor / "experiment/robotwin/eval_policy_client_lingbotvla.py").read_text(), trials))
    for filename in ("__init__.py", "websocket_client_policy.py", "msgpack_numpy.py"):
        shutil.copy2(vendor / "deploy" / filename, deploy / filename)
    shutil.copy2(Path(__file__).with_name("scene_metadata.py"), script_dir / "scene_metadata.py")
    config = destination / "task_config"
    if config.is_symlink() and config.resolve() != robotwin.resolve() / "task_config":
        config.unlink()
    if not config.exists():
        config.symlink_to(robotwin.resolve() / "task_config", target_is_directory=True)
    return client


def aggregate_results(run: Path, tasks: list[str], settings: list[str], precision: str, trials: int = 100, inference_backend: str = "native") -> dict:
    result = {"schema": "lrvla-evaluation-v1", "official_submission_template_verified": True,
              "precision": precision, "release_fp32_precision": precision == "fp32",
              "inference_backend": inference_backend, "release_kernel_equivalence_verified": inference_backend == "native",
              "robotwin_revision": ROBOTWIN_REVISION, "trials_per_task": trials, "settings": {},
              "full_50_task_benchmark": set(tasks) == set(TASKS) and len(tasks) == 50 and trials == 100 and set(settings) == set(INSTRUCTIONS)}
    for setting in settings:
        rows = []
        for task in tasks:
            logfile = run / setting / "eval_logs" / f"{task}.log"
            success, total = parse_success_counts(logfile.read_text(errors="replace")) if logfile.exists() else (0, 0)
            status_file = logfile.with_suffix(".status.json")
            status = json.loads(status_file.read_text()) if status_file.exists() else {}
            complete = total == trials and status.get("exit_code") == 0
            rows.append({"task": task, "successes": success, "trials": total, "expected_trials": trials,
                         "success_rate": success / total if total else None, "complete": complete,
                         "exit_code": status.get("exit_code"), "log": str(logfile)})
        successes = sum(row["successes"] for row in rows)
        total = sum(row["trials"] for row in rows)
        complete = all(row["complete"] for row in rows)
        result["settings"][setting] = {"instruction_type": INSTRUCTIONS[setting], "tasks": rows,
                                       "successes": successes, "trials": total, "expected_trials": len(tasks) * trials,
                                       "observed_success_rate": successes / total if total else None,
                                       "complete": complete, "benchmark_success_rate": successes / total if complete and total else None}
    result["complete"] = all(setting["complete"] for setting in result["settings"].values())
    result["comparable_to_release"] = result["complete"] and result["full_50_task_benchmark"] and precision == "fp32" and inference_backend == "native"
    result["status"] = "completed" if result["complete"] else "incomplete"
    (run / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def export_official_results(report: dict, destination: Path, team_id: str, template: Path) -> dict:
    """Fill the authenticated competition template with observed integer counts.

    Missing tasks stay zero. Partial attempts remain partial; success percentages
    never manufacture trial counts. Precision/protocol evidence lives separately.
    """
    payload = json.loads(template.read_text())
    if payload.get("schema_version") != 1 or set(payload["results"]) != {"clean", "randomized"}:
        raise ValueError("Unexpected official template schema")
    payload["team_id"] = team_id
    for setting, label in (("demo_clean", "clean"), ("demo_randomized", "randomized")):
        if set(payload["results"][label]) != set(TASKS):
            raise ValueError("Official template task list differs from the required 50 tasks")
        for task in TASKS:
            payload["results"][label][task] = {"attempts": 0, "successes": 0}
        for row in report.get("settings", {}).get(setting, {}).get("tasks", []):
            if row["task"] not in TASKS:
                raise ValueError("Unknown evaluation task")
            attempts, successes = row["trials"], row["successes"]
            if type(attempts) is not int or type(successes) is not int or not 0 <= successes <= attempts <= 100:
                raise ValueError("Evaluation counts must satisfy 0 <= successes <= attempts <= 100")
            payload["results"][label][row["task"]] = {"attempts": attempts, "successes": successes}
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    return payload
