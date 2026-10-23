"""Fixed 50 tasks x 2 settings x 10 scenes RoboTwin screening on local GPUs.

Every run evaluates the same 1,000 scenes: all 50 tasks, demo_clean with seen
and demo_randomized with unseen instructions, the first 10 scene seeds of each.
Each GPU shard runs the unchanged scripts/evaluate.py with one policy server
and one or more simulators. Parts of the task list may run on other machines;
--collect merges them into one report.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import queue
import re
import signal
import subprocess
import sys
import threading
import time

import yaml

from .data import TASKS
from .evaluation import aggregate_results, evaluation_task_complete, export_official_results, parse_success_counts

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "lrvla-quick-evaluation-v1"
SETTINGS = ("demo_clean", "demo_randomized")
LABELS = {"demo_clean": "clean", "demo_randomized": "randomized"}
TRIALS = 10
# A100-SXM4-40GB: the FP32 policy server alone used 24.9 GiB; with one simulator
# it peaked at 34.0 GiB. Further simulators are assumed to add the same 9 GiB.
FP32_SERVER_MIB = 25 * 1024
SIMULATOR_MIB = 9 * 1024
PROFILE_FIELDS = {"backend", "precision", "use_length", "num_steps", "compile", "action_smoothing",
                  "sim_workers", "policy_rng", "expert_cache", "expert_cache_mode"}
# Scene-keyed and process-global policy noise measure the same policy; parts
# using either can be collected together, everything else must match.
PROTOCOL_KEYS = ("precision", "inference_backend", "use_length", "num_steps", "compile", "action_smoothing",
                 "expert_seed_policy", "expert_cache_mode", "trials", "settings")
PLAN_KEYS = ("schema", "trials", "settings", "checkpoint", "config", "precision", "part", "part_tasks")
METRIC_PHASES = ("expert_seconds", "policy_setup_seconds", "instruction_seconds", "observation_seconds",
                 "server_infer_seconds", "rpc_wait_and_transport_seconds", "action_seconds", "total_seconds")
METRICS = re.compile(r"LRVLA_EPISODE_METRICS=(\{[^\r\n]*\})")
TEMPLATE = ROOT / "docs/official-templates/初赛评测结果模板_JSON版.json"


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def write_json(path, payload):
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def parse_part(value):
    match = re.fullmatch(r"(\d+)/(\d+)", value)
    if not match or not 1 <= int(match[1]) <= int(match[2]) <= len(TASKS):
        raise argparse.ArgumentTypeError("expected I/N with 1 <= I <= N <= 50")
    return int(match[1]), int(match[2])


def resolve_checkpoint(path):
    """Accept .../checkpoints/global_step_N/hf_ckpt or a model directory holding exactly one export."""
    path = Path(path).expanduser().resolve()
    if not any(path.glob("*.safetensors")):
        exports = sorted(path.glob("checkpoints/global_step_*/hf_ckpt"))
        if len(exports) != 1:
            raise FileNotFoundError(f"Expected one exported hf_ckpt under {path}, found {len(exports)}; "
                                    "pass .../checkpoints/global_step_N/hf_ckpt.")
        path = exports[0]
        if not any(path.glob("*.safetensors")):
            raise FileNotFoundError(f"No safetensors shards in {path}")
    run = path.parent.parent.parent
    for required in ("lingbotvla_cli.yaml", "configs/robot_configs/robotwin.yaml"):
        if not (run / required).is_file():
            raise FileNotFoundError(f"Download the whole model directory; missing {run / required}")
    return path


def default_output(checkpoint, part, precision):
    step = re.fullmatch(r"global_step_(\d+)", checkpoint.parent.name)
    name = f"quick-{checkpoint.parent.parent.parent.name}-step{step[1] if step else checkpoint.parent.name}"
    if precision != "fp32":
        name += f"-{precision}"
    if part[1] > 1:
        name += f"-part{part[0]}of{part[1]}"
    return ROOT / "runs/evaluation" / name


def load_profile(config):
    profile = yaml.safe_load(Path(config).read_text()) or {}
    if not isinstance(profile, dict) or set(profile) - PROFILE_FIELDS:
        raise ValueError(f"{config} is not an inference profile like configs/eval.yaml.")
    if profile.get("sim_workers", 1) != 1 or profile.get("expert_cache_mode", "off") != "off":
        raise ValueError("Quick evaluation runs one simulator per GPU and no expert cache.")
    return profile


def step_limits(robotwin):
    limits = yaml.safe_load((Path(robotwin) / "task_config/_eval_step_limit.yml").read_text()) or {}
    if not set(TASKS) <= set(limits):
        raise ValueError("RoboTwin step limits do not cover the 50 evaluation tasks.")
    return {task: int(limits[task]) for task in TASKS}


def balanced_split(tasks, count, cost):
    """Longest-first greedy split by step limit; deterministic for the same inputs."""
    if not 1 <= count <= len(tasks):
        raise ValueError("Cannot split tasks into empty groups.")
    groups, loads = [[] for _ in range(count)], [0] * count
    for task in sorted(tasks, key=lambda name: (-cost[name], name)):
        index = min(range(count), key=lambda position: (loads[position], position))
        groups[index].append(task)
        loads[index] += cost[task]
    return [sorted(group) for group in groups]


def build_plan(checkpoint, part, gpus, limits, config, precision, sims_per_gpu=1):
    part_tasks = balanced_split(list(TASKS), part[1], limits)[part[0] - 1]
    shards = balanced_split(part_tasks, min(gpus, len(part_tasks)), limits)
    # Shared policy servers need per-scene noise streams; one simulator keeps
    # the evaluator's default process-global stream.
    return {"schema": SCHEMA, "trials": TRIALS, "settings": list(SETTINGS), "checkpoint": str(checkpoint),
            "config": str(Path(config).resolve()), "precision": precision, "part": list(part),
            "part_tasks": part_tasks, "sims_per_gpu": sims_per_gpu,
            "policy_rng": "legacy" if sims_per_gpu == 1 else "scene",
            "shards": [{"name": f"s{index}", "tasks": tasks} for index, tasks in enumerate(shards)]}


def reuse_plan(output, plan):
    """Resume with the saved shard layout; evaluate.py keeps each shard's task list fixed."""
    saved = output / "quick_plan.json"
    if not saved.is_file():
        return plan
    previous = json.loads(saved.read_text())
    for key in PLAN_KEYS:
        if previous.get(key) != plan[key]:
            raise ValueError(f"{saved} was planned with a different {key}; repeat the original arguments "
                             "or choose a new --output.")
    if previous["policy_rng"] == "legacy" and plan["sims_per_gpu"] > 1:
        raise ValueError(f"{saved} started with one simulator per GPU; resume with --sims-per-gpu 1 "
                         "or choose a new --output.")
    return {**previous, "sims_per_gpu": plan["sims_per_gpu"]}


def shard_dir(output, shard):
    return Path(output) / "shards" / shard["name"]


def shard_complete(output, shard):
    return all(evaluation_task_complete(shard_dir(output, shard), setting, task, TRIALS)
               for setting in SETTINGS for task in shard["tasks"])


def evaluator_arguments(plan, shard, output, gpu, port):
    return ["--config", plan["config"], "--checkpoint", plan["checkpoint"], "--tasks", ",".join(shard["tasks"]),
            "--setting", "both", "--trials", str(TRIALS), "--precision", plan["precision"],
            "--output", str(shard_dir(output, shard)), "--gpu", gpu, "--port", str(port), "--skip-completed",
            "--sim-workers", str(plan["sims_per_gpu"]), "--policy-rng", plan["policy_rng"]]


def gpu_inventory(selectors):
    """Resolve nvidia-smi indices or UUIDs to physical GPUs with their memory in MiB."""
    from .parallel_evaluation import assigned_devices
    from .wsl_tools import linux_executable
    executable = linux_executable("nvidia-smi")
    devices = assigned_devices(selectors)
    for device in devices:
        row = subprocess.run([str(executable), "--query-gpu=name,memory.total,memory.used",
                              "--format=csv,noheader,nounits", "-i", device["uuid"]],
                             capture_output=True, text=True, check=True).stdout.strip()
        name, total, used = (part.strip() for part in row.rsplit(",", 2))
        device.update(name=name, total_mib=int(float(total)), used_mib=int(float(used)))
    return devices


def check_memory(devices, precision, sims_per_gpu=1):
    if precision != "fp32":
        return
    needed = FP32_SERVER_MIB + sims_per_gpu * SIMULATOR_MIB
    short = [device for device in devices if device["total_mib"] - device["used_mib"] < needed]
    if short:
        details = ", ".join(f"{device['name']} ({device['uuid']}) has {(device['total_mib'] - device['used_mib']) / 1024:.1f} GiB free"
                            for device in short)
        raise RuntimeError(f"An FP32 policy server with {sims_per_gpu} simulator(s) needs about {needed / 1024:.0f} GiB; "
                           f"{details}. Use fewer --sims-per-gpu, GPUs with more free memory, --precision bf16 "
                           "(approximate) or --no-memory-check.")


def progress(output, plan, limits):
    """Scenes finished so far from the evaluator logs, plus step-limit-weighted work."""
    counts, work = {setting: [0, 0] for setting in SETTINGS}, 0
    for shard in plan["shards"]:
        for setting in SETTINGS:
            for task in shard["tasks"]:
                log = shard_dir(output, shard) / setting / "eval_logs" / f"{task}.log"
                try:
                    successes, scenes = parse_success_counts(log.read_text(errors="replace"))
                except (OSError, ValueError):
                    continue
                counts[setting][0] += successes
                counts[setting][1] += scenes
                work += scenes * limits[task]
    return counts, work


def progress_line(counts, total_scenes, elapsed, eta):
    parts = [f"{sum(scenes for _, scenes in counts.values())}/{total_scenes} scenes"]
    for setting in SETTINGS:
        successes, scenes = counts[setting]
        parts.append(f"{LABELS[setting]} {successes}/{scenes}" + (f" ({100 * successes / scenes:.0f}%)" if scenes else ""))
    parts.append(f"elapsed {elapsed / 3600:.1f} h")
    if eta is not None:
        parts.append(f"ETA {eta / 3600:.1f} h")
    return "[quick-eval] " + " | ".join(parts)


def run_shards(plan, output, devices, *, bind, retries, limits, progress_interval, evaluate_script,
               python=sys.executable):
    """Run unfinished shards over the GPUs, one evaluator process per GPU at a time."""
    from .parallel_evaluation import available_port
    pending = queue.Queue()
    for shard in plan["shards"]:
        if not shard_complete(output, shard):
            pending.put(shard)
    (output / "logs").mkdir(parents=True, exist_ok=True)
    status_file = output / "status.json"
    attempts = json.loads(status_file.read_text()) if status_file.is_file() else []
    processes, ports, lock, stopped = {}, set(), threading.Lock(), threading.Event()

    def reserve_port():
        with lock:
            for _ in range(100):
                port = available_port()
                if port not in ports:
                    ports.add(port)
                    return port
        raise RuntimeError("Could not allocate a distinct inference port.")

    def launch(shard, device, attempt):
        port = reserve_port()
        command = [str(python), str(evaluate_script), *evaluator_arguments(plan, shard, output, device["uuid"], port)]
        env = dict(os.environ, PYTHONUNBUFFERED="1")
        env.pop("LRVLA_RENDER_PCI", None)
        if bind:
            # Several GPUs: render each simulator on its policy GPU, as evaluate.py
            # does for shared servers; CUDA visibility alone does not pick Vulkan.
            env["LRVLA_RENDER_PCI"] = device["pci"]
        record = {"shard": shard["name"], "attempt": attempt, "gpu": device["uuid"], "port": port,
                  "render_pci": env.get("LRVLA_RENDER_PCI"), "log": str(output / "logs" / f"{shard['name']}.log"),
                  "started_at": now()}
        with open(record["log"], "a") as log:
            log.write(f"\n=== {record['started_at']} attempt {attempt} on {device['uuid']} port {port}\n")
            log.flush()
            with lock:
                if stopped.is_set():
                    return None
                process = subprocess.Popen(command, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                                           start_new_session=True)
                processes[process.pid] = process
            code = process.wait()
            with lock:
                processes.pop(process.pid, None)
        record.update(exit_code=code, finished_at=now())
        with lock:
            attempts.append(record)
            write_json(status_file, attempts)
        print(f"[quick-eval] shard {shard['name']} attempt {attempt} on {device['uuid']}: exit {code}", flush=True)
        return code

    def worker(device):
        while not stopped.is_set():
            try:
                shard = pending.get_nowait()
            except queue.Empty:
                return
            for attempt in range(1, retries + 2):
                code = launch(shard, device, attempt)
                if code in (0, None) or stopped.is_set():
                    break

    def terminate(signum, frame):
        stopped.set()
        with lock:
            for process in list(processes.values()):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    total_scenes = len(plan["part_tasks"]) * len(SETTINGS) * TRIALS
    total_work = sum(limits[task] for task in plan["part_tasks"]) * len(SETTINGS) * TRIALS
    counts, start_work = progress(output, plan, limits)
    started = time.monotonic()
    print(progress_line(counts, total_scenes, 0, None), flush=True)
    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = {pool.submit(worker, device) for device in devices}
            while futures:
                finished, futures = wait(futures, timeout=progress_interval, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                counts, work = progress(output, plan, limits)
                elapsed = time.monotonic() - started
                rate = (work - start_work) / elapsed if elapsed > 0 else 0
                print(progress_line(counts, total_scenes, elapsed, (total_work - work) / rate if rate > 0 else None),
                      flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return not stopped.is_set()


def load_counts(path):
    """Per-setting {task: (successes, scenes)} from results.json, official_results.json or a quick directory."""
    path = Path(path)
    if path.is_dir():
        path = path / "results.json"
    payload = json.loads(path.read_text())
    counts = {setting: {} for setting in SETTINGS}
    if payload.get("schema") == "lrvla-evaluation-v1":
        for setting in SETTINGS:
            for row in payload.get("settings", {}).get(setting, {}).get("tasks", []):
                counts[setting][row["task"]] = (row["successes"], row["trials"])
    elif payload.get("schema_version") == 1 and isinstance(payload.get("results"), dict):
        for setting in SETTINGS:
            for task, row in payload["results"].get(LABELS[setting], {}).items():
                counts[setting][task] = (row["successes"], row["attempts"])
    else:
        raise ValueError(f"Unrecognized evaluation result file: {path}")
    for setting in SETTINGS:
        for task, (successes, scenes) in list(counts[setting].items()):
            if task not in TASKS or type(successes) is not int or type(scenes) is not int or not 0 <= successes <= scenes:
                raise ValueError(f"Invalid counts for {LABELS[setting]}/{task} in {path}")
            if not scenes:
                del counts[setting][task]
    return counts


def setting_summary(counts):
    """Success rate and a 95% half-width, treating tasks as independent binomial strata."""
    scenes = sum(total for _, total in counts.values())
    successes = sum(success for success, _ in counts.values())
    if not scenes:
        return {"successes": 0, "scenes": 0, "rate": None, "ci95": None}
    variance = sum(success * (total - success) / total for success, total in counts.values())
    return {"successes": successes, "scenes": scenes, "rate": successes / scenes,
            "ci95": 1.96 * math.sqrt(variance) / scenes}


def mean_summary(parts):
    if any(part["rate"] is None for part in parts):
        return {"rate": None, "ci95": None}
    return {"rate": sum(part["rate"] for part in parts) / len(parts),
            "ci95": math.sqrt(sum(part["ci95"] ** 2 for part in parts)) / len(parts)}


def difference(current, baseline):
    if current["rate"] is None or baseline["rate"] is None:
        return None
    return {"delta": current["rate"] - baseline["rate"], "ci95": math.hypot(current["ci95"], baseline["ci95"])}


def time_breakdown(logs):
    """Mean per-scene seconds of each stage from the evaluator's LRVLA_EPISODE_METRICS lines."""
    totals, episodes = dict.fromkeys(METRIC_PHASES, 0.0), 0
    for log in logs:
        try:
            text = Path(log).read_text(errors="replace")
        except OSError:
            continue
        for match in METRICS.finditer(text):
            try:
                data = json.loads(match[1])
            except ValueError:
                continue
            episodes += 1
            for key in totals:
                totals[key] += float(data.get(key) or 0)
    if not episodes:
        return None
    return {"episodes": episodes, **{key: value / episodes for key, value in totals.items()}}


def summarize(report, *, profile, checkpoints, identity, policy_rng=(), baseline=None, baseline_source=None):
    rows = {setting: {row["task"]: row for row in report["settings"][setting]["tasks"]} for setting in SETTINGS}
    counts = {setting: {task: (row["successes"], row["trials"]) for task, row in rows[setting].items() if row["trials"]}
              for setting in SETTINGS}
    settings = {setting: setting_summary(counts[setting]) for setting in SETTINGS}
    summary = {"schema": SCHEMA + "-summary", "protocol": "50 tasks x 2 settings x 10 scenes",
               "complete": report["complete"], "checkpoints": checkpoints, "weights_sha256": identity,
               "profile": profile, "policy_rng": sorted(policy_rng),
               "expected_scenes": len(TASKS) * len(SETTINGS) * TRIALS,
               "scenes": sum(part["scenes"] for part in settings.values()),
               "settings": settings, "mean": mean_summary(list(settings.values())),
               "incomplete": [f"{LABELS[setting]}/{task}" for setting in SETTINGS
                              for task, row in rows[setting].items() if not row["complete"]],
               "time_per_scene": time_breakdown(row["log"] for setting in SETTINGS for row in rows[setting].values()),
               "tasks": [{"task": task, **{LABELS[setting]: list(counts[setting][task]) if task in counts[setting] else None
                                           for setting in SETTINGS}} for task in TASKS]}
    if baseline is not None:
        # Compare only finished task/setting pairs that the baseline also covers.
        matched = {setting: [task for task, row in rows[setting].items() if row["complete"] and task in baseline[setting]]
                   for setting in SETTINGS}
        current = {setting: setting_summary({task: counts[setting][task] for task in matched[setting]}) for setting in SETTINGS}
        base = {setting: setting_summary({task: baseline[setting][task] for task in matched[setting]}) for setting in SETTINGS}
        current_mean, base_mean = mean_summary(list(current.values())), mean_summary(list(base.values()))
        summary["baseline"] = {"source": baseline_source, "matched_pairs": sum(len(tasks) for tasks in matched.values()),
                               "settings": base, "mean": base_mean,
                               "difference": {**{setting: difference(current[setting], base[setting]) for setting in SETTINGS},
                                              "mean": difference(current_mean, base_mean)}}
        for row in summary["tasks"]:
            row.update({f"baseline_{LABELS[setting]}": list(baseline[setting][row["task"]])
                        if row["task"] in baseline[setting] else None for setting in SETTINGS})
    return summary


def collect(directories, destination, baseline=None):
    """Merge quick evaluation directories (one per machine or part) into one report."""
    destination = Path(destination).resolve()
    destination.mkdir(parents=True, exist_ok=True)
    task_runs, checkpoints, identities, rngs, profile, reference = {}, [], set(), set(), None, None
    for directory in (Path(value).resolve() for value in directories):
        quick = json.loads((directory / "quick_plan.json").read_text())
        if quick.get("schema") != SCHEMA or quick.get("trials") != TRIALS or quick.get("settings") != list(SETTINGS):
            raise ValueError(f"{directory} is not a 50 x 2 x 10 quick evaluation.")
        checkpoints.append(quick["checkpoint"])
        for shard in quick["shards"]:
            run = shard_dir(directory, shard)
            plan_file = run / "evaluation_plan.json"
            if plan_file.is_file():
                evaluation = json.loads(plan_file.read_text())
                current = {key: evaluation.get(key) for key in PROTOCOL_KEYS}
                if profile is None:
                    profile, reference = current, evaluation
                elif current != profile:
                    raise ValueError(f"{run} used a different inference profile; collect runs of one protocol only.")
                rngs.add(evaluation.get("policy_rng") or "legacy")
                if evaluation.get("checkpoint_identity"):
                    identities.add(evaluation["checkpoint_identity"]["sha256"])
            for task in shard["tasks"]:
                if task in task_runs:
                    raise ValueError(f"{task} appears in more than one collected evaluation.")
                task_runs[task] = run
    if profile is None:
        raise ValueError("No evaluation has started in the given directories.")
    if len(identities) > 1:
        raise ValueError("The collected evaluations used different checkpoint weights.")
    for task in TASKS:
        task_runs.setdefault(task, destination / "not-collected")
    report = aggregate_results(destination, list(TASKS), list(SETTINGS), profile["precision"], TRIALS,
                               profile["inference_backend"], task_runs=task_runs, use_length=profile["use_length"],
                               expert_seed_policy=profile["expert_seed_policy"], inference_profile=reference)
    export_official_results(report, destination / "official_results.json", "replace_with_your_team_id", TEMPLATE)
    summary = summarize(report, profile=profile, checkpoints=sorted(set(checkpoints)),
                        identity=next(iter(identities), None), policy_rng=rngs,
                        baseline=load_counts(baseline) if baseline else None,
                        baseline_source=str(Path(baseline).resolve()) if baseline else None)
    write_json(destination / "summary.json", summary)
    (destination / "summary.md").write_text(render_summary(summary))
    return summary


def percent(value):
    return "—" if value is None else f"{100 * value:.1f}%"


def points(value, signed=False):
    return "—" if value is None else (f"{100 * value:+.1f}" if signed else f"±{100 * value:.1f}")


def fraction(pair):
    return "—" if pair is None else f"{pair[0]}/{pair[1]}"


def render_summary(summary):
    profile = summary["profile"]
    lines = [f"# 快速评测：{'，'.join(summary['checkpoints'])}", "",
             "- 协议：50 任务 × 2 设定 × 10 场景；clean 使用 seen 指令，randomized 使用 unseen 指令，"
             "每个任务-设定固定使用前 10 个场景种子",
             f"- 推理：{profile['inference_backend']} {profile['precision'].upper()}，执行 {profile['use_length']} 步，"
             f"去噪 {profile['num_steps']} 步，compile {'开' if profile['compile'] else '关'}，"
             f"policy_rng {'/'.join(summary['policy_rng']) or '—'}",
             f"- 权重 sha256：{summary['weights_sha256'] or '未记录'}",
             f"- 进度：{summary['scenes']}/{summary['expected_scenes']} 场景，{'已完成' if summary['complete'] else '未完成'}"]
    baseline = summary.get("baseline")
    if baseline:
        lines.append(f"- 对照：{baseline['source']}，按双方都有结果的 {baseline['matched_pairs']} 个任务-设定组合比较")
    header = "| 设定 | 成功/场景 | 成功率 | 95% 区间 |" + (" 对照 | 差值 | 差值 95% 区间 |" if baseline else "")
    lines += ["", header, "|---|---:|---:|---:|" + ("---:|---:|---:|" if baseline else "")]
    for key, name in (("demo_clean", "clean（seen）"), ("demo_randomized", "randomized（unseen）"), ("mean", "两设定平均")):
        part = summary["mean"] if key == "mean" else summary["settings"][key]
        cells = [name, f"{part['successes']}/{part['scenes']}" if key != "mean" else "—",
                 percent(part["rate"]), points(part["ci95"])]
        if baseline:
            base = baseline["mean"] if key == "mean" else baseline["settings"][key]
            change = baseline["difference"][key]
            cells += [percent(base["rate"]), points(change and change["delta"], signed=True),
                      points(change and change["ci95"])]
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "每个任务只有 10 个场景，单任务成功率的 95% 区间约 ±30 个百分点；比较模型时看两设定平均及其区间。"]
    if summary["incomplete"]:
        lines.append(f"未完成：{', '.join(summary['incomplete'])}")
    if baseline:
        changes = []
        for row in summary["tasks"]:
            pairs = [(row[label], row[f"baseline_{label}"]) for label in LABELS.values()]
            if all(current and base and current[1] == TRIALS for current, base in pairs):
                changes.append((sum(current[0] / current[1] - base[0] / base[1] for current, base in pairs), row["task"]))
        changes.sort()
        if changes:
            lines += ["", "两设定合计变化最大的任务（百分点）：",
                      "- 提升：" + "，".join(f"{task} {100 * value:+.0f}" for value, task in changes[::-1][:5] if value > 0),
                      "- 下降：" + "，".join(f"{task} {100 * value:+.0f}" for value, task in changes[:5] if value < 0)]
    timing = summary["time_per_scene"]
    if timing:
        names = (("expert_seconds", "专家预检"), ("policy_setup_seconds", "场景初始化"), ("instruction_seconds", "生成指令"),
                 ("observation_seconds", "取观测"), ("server_infer_seconds", "模型推理"),
                 ("rpc_wait_and_transport_seconds", "通信与排队"), ("action_seconds", "仿真执行动作"),
                 ("total_seconds", "合计"))
        lines += ["", f"## 每场景平均耗时（秒，{timing['episodes']} 个场景）", "",
                  "| " + " | ".join(name for _, name in names) + " |", "|" + "---:|" * len(names),
                  "| " + " | ".join(f"{timing[key]:.1f}" for key, _ in names) + " |"]
    lines += ["", "## 各任务", "",
              "| 任务 | clean | randomized |" + (" 对照 clean | 对照 randomized |" if baseline else ""),
              "|---|---:|---:|" + ("---:|---:|" if baseline else "")]
    for row in summary["tasks"]:
        cells = [row["task"], fraction(row["clean"]), fraction(row["randomized"])]
        if baseline:
            cells += [fraction(row["baseline_clean"]), fraction(row["baseline_randomized"])]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", type=Path,
                        help="Exported .../checkpoints/global_step_N/hf_ckpt, or the model directory holding one export")
    parser.add_argument("--gpus", default="0",
                        help="Comma-separated nvidia-smi indices or UUIDs; each GPU runs one policy server")
    parser.add_argument("--sims-per-gpu", type=int, default=1,
                        help="Simulators sharing each GPU's policy server; FP32 needs about 25 + 9 x N GiB")
    parser.add_argument("--part", type=parse_part, default=(1, 1), metavar="I/N",
                        help="Run part I of N of the 50 tasks, e.g. 1/2 and 2/2 on two machines; merge with --collect")
    parser.add_argument("--output", type=Path,
                        help="Defaults to runs/evaluation/quick-<model>-step<N>; rerun the same command to resume")
    parser.add_argument("--config", type=Path, default=ROOT / "configs/eval.yaml",
                        help="Inference profile passed to scripts/evaluate.py")
    parser.add_argument("--precision", choices=["fp32", "bf16"],
                        help="Override the profile; BF16 is approximate, for GPUs with less than 34 GiB free")
    parser.add_argument("--baseline", type=Path,
                        help="results.json, official_results.json or a quick evaluation directory to compare with")
    parser.add_argument("--collect", type=Path, nargs="+", metavar="DIR",
                        help="Merge and summarize quick evaluation directories without running anything")
    parser.add_argument("--retries", type=int, default=1, help="Restarts of a failed GPU shard; finished tasks are kept")
    parser.add_argument("--progress-interval", type=int, default=300, help="Seconds between progress lines")
    parser.add_argument("--no-memory-check", action="store_true")
    parser.add_argument("--dry-run", action="store_true", help="Write the plans without loading the model or starting simulators")
    parser.add_argument("--evaluate-script", type=Path, default=ROOT / "scripts/evaluate.py", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.collect:
        if args.checkpoint or args.dry_run:
            parser.error("--collect only merges existing quick evaluation directories")
        if len(args.collect) > 1 and not args.output:
            parser.error("--collect with several directories needs --output")
        return args
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    args.gpu_list = [value.strip() for value in args.gpus.split(",") if value.strip()]
    if not args.gpu_list or len(set(args.gpu_list)) != len(args.gpu_list):
        parser.error("--gpus needs distinct GPU selectors")
    if args.retries < 0 or args.progress_interval < 1 or not 1 <= args.sims_per_gpu <= 8:
        parser.error("--retries must be >= 0, --progress-interval >= 1 and --sims-per-gpu 1..8")
    if os.environ.get("PBS_JOBID") and (len(args.gpu_list) > 1 or args.sims_per_gpu > 1):
        parser.error("Parallel simulator evaluation took cluster compute nodes down; inside PBS jobs use one GPU "
                     "and one simulator.")
    return args


def main(argv=None):
    args = parse_args(argv)
    if args.collect:
        summary = collect(args.collect, args.output or args.collect[0], args.baseline)
        print(render_summary(summary))
        return 0 if summary["complete"] else 1
    checkpoint = resolve_checkpoint(args.checkpoint)
    precision = args.precision or load_profile(args.config).get("precision", "fp32")
    limits = step_limits(ROOT / "vendor/RoboTwin")
    output = (args.output or default_output(checkpoint, args.part, precision)).resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan = reuse_plan(output, build_plan(checkpoint, args.part, len(args.gpu_list), limits, args.config, precision,
                                         args.sims_per_gpu))
    write_json(output / "quick_plan.json", plan)
    if args.dry_run:
        devices = [{"uuid": value, "pci": None, "name": "dry-run"} for value in args.gpu_list]
    else:
        devices = gpu_inventory(",".join(args.gpu_list))
        if not args.no_memory_check:
            check_memory(devices, precision, args.sims_per_gpu)
    print(f"[quick-eval] checkpoint {checkpoint}", flush=True)
    print(f"[quick-eval] output {output}", flush=True)
    print(f"[quick-eval] part {plan['part'][0]}/{plan['part'][1]}: {len(plan['part_tasks'])} tasks x 2 settings x {TRIALS} "
          f"scenes, {precision}, {args.sims_per_gpu} simulator(s) per GPU, {len(plan['shards'])} shard(s) over "
          + ", ".join(f"{device['uuid']} ({device['name']})" for device in devices), flush=True)
    if args.sims_per_gpu > 1 and (os.cpu_count() or 0) < len(devices) * (args.sims_per_gpu + 1):
        print(f"[quick-eval] warning: {os.cpu_count()} CPU cores for {len(devices) * args.sims_per_gpu} simulators; "
              "each simulator needs about one core.", flush=True)
    if args.dry_run:
        for index, shard in enumerate(plan["shards"]):
            command = [sys.executable, str(args.evaluate_script),
                       *evaluator_arguments(plan, shard, output, devices[index % len(devices)]["uuid"], 9330 + index),
                       "--dry-run"]
            subprocess.run(command, cwd=ROOT, check=True)
        return 0
    finished = run_shards(plan, output, devices, bind=len(devices) > 1, retries=args.retries, limits=limits,
                          progress_interval=args.progress_interval, evaluate_script=args.evaluate_script.resolve())
    try:
        summary = collect([output], output, args.baseline)
    except ValueError as error:
        print(f"[quick-eval] no results to summarize ({error}); read {output / 'logs'}", flush=True)
        return 1
    print(render_summary(summary), flush=True)
    print(f"[quick-eval] summary {output / 'summary.md'}; results {output / 'results.json'}", flush=True)
    if not finished:
        return 130
    return 0 if summary["complete"] else 1
