"""Policy servers on one machine, RoboTwin simulators on another: fixed 50 x 2 x 10 screening.

`serve` starts one unchanged native policy server per GPU slot and writes a manifest with
the host, ports, inference profile and checkpoint identity once every server accepts
connections. `evaluate` runs simulator clients against already reachable endpoints (for
example SSH-forwarded local ports), one client per endpoint at a time, pulling task/setting
pairs from a shared queue. Results use the quick-evaluation layout, so its summary,
baseline comparison and --collect apply unchanged. How endpoints are made reachable is
site specific and stays outside this module.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

from .data import TASKS
from .evaluation import EXPERT_SEED_POLICY, INSTRUCTIONS, create_eval_runtime, evaluation_task_complete
from . import quick_evaluation as quick

ROOT = Path(__file__).resolve().parents[2]
MANIFEST_SCHEMA = "lrvla-remote-policy-servers-v1"
PROFILE = {"precision": "bf16", "inference_backend": "native", "use_length": 50, "num_steps": 10,
           "compile": False, "action_smoothing": "none", "policy_rng": "legacy"}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def port_open(host, port, timeout=2.0):
    try:
        with socket.create_connection((host, port), timeout=timeout):
            return True
    except OSError:
        return False


def parse_endpoints(value):
    endpoints = []
    for item in value.split(","):
        host, _, port = item.strip().rpartition(":")
        if not host or not port.isdigit():
            raise argparse.ArgumentTypeError("expected host:port[,host:port...]")
        endpoints.append((host, int(port)))
    if len(set(endpoints)) != len(endpoints):
        raise argparse.ArgumentTypeError("endpoints must be distinct")
    return endpoints


# ----------------------------------------------------------------------------- servers

def server_command(checkpoint, port, *, python=sys.executable, script=ROOT / "scripts/serve_native.py"):
    precision = PROFILE["precision"]
    return [str(python), str(script), "--model_path", str(checkpoint), "--use_length", str(PROFILE["use_length"]),
            "--use_bf16", str(precision == "bf16"), "--use_fp32", str(precision == "fp32"),
            "--use_compile", str(PROFILE["compile"]), "--port", str(port), "--num_steps", str(PROFILE["num_steps"])]


def serve(checkpoint, manifest, ports, gpus, *, qwen=ROOT / "models/Qwen3-VL-4B-Instruct", timeout=1800,
          script=ROOT / "scripts/serve_native.py", python=sys.executable, identity=True):
    """Start one server per (port, gpu); write the manifest when all are ready; stay up until signalled."""
    from .training_identity import base_checkpoint_identity
    checkpoint = Path(checkpoint).resolve()
    run = checkpoint.parent.parent.parent
    if len(ports) != len(gpus):
        raise ValueError("Give one GPU per port.")
    logs = Path(manifest).with_suffix(".logs")
    logs.mkdir(parents=True, exist_ok=True)
    processes = []
    stopping = threading.Event()

    def terminate(signum, frame):
        stopping.set()

    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        for port, gpu in zip(ports, gpus):
            env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(gpu), QWEN3VL_PATH=str(Path(qwen).resolve()),
                       PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1", TOKENIZERS_PARALLELISM="false")
            processes.append(subprocess.Popen(server_command(checkpoint, port, python=python, script=script), cwd=run, env=env,
                                              stdout=open(logs / f"server-{port}.log", "w"), stderr=subprocess.STDOUT,
                                              start_new_session=True))
        deadline = time.monotonic() + timeout
        while not all(port_open("127.0.0.1", port) for port in ports):
            if stopping.is_set() or time.monotonic() > deadline or any(p.poll() is not None for p in processes):
                raise RuntimeError(f"Policy servers did not all become ready; read {logs}")
            time.sleep(5)
        record = {"schema": MANIFEST_SCHEMA, "host": socket.gethostname(), "started_at": now(),
                  "checkpoint": str(checkpoint), "checkpoint_identity": base_checkpoint_identity(checkpoint) if identity else None,
                  "endpoints": [{"port": port, "gpu": str(gpu)} for port, gpu in zip(ports, gpus)], **PROFILE}
        quick.write_json(Path(manifest), record)
        print(json.dumps({key: record[key] for key in ("host", "checkpoint", "endpoints")}), flush=True)
        while not stopping.is_set():
            dead = [port for port, process in zip(ports, processes) if process.poll() is not None]
            if dead:
                raise RuntimeError(f"Policy servers exited on ports {dead}; read {logs}")
            time.sleep(5)
        return 0
    finally:
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
        for process in processes:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
        for sig, handler in previous.items():
            signal.signal(sig, handler)


# ----------------------------------------------------------------------------- simulators

def load_manifest(path):
    record = json.loads(Path(path).read_text())
    if record.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"{path} is not a policy server manifest.")
    for key, value in PROFILE.items():
        if record.get(key) != value:
            raise ValueError(f"Server profile differs from the quick protocol: {key}={record.get(key)!r}")
    return record


def evaluation_plan(record, endpoints):
    return {"checkpoint": record["checkpoint"], "checkpoint_identity": record.get("checkpoint_identity"),
            "precision": record["precision"], "inference_backend": record["inference_backend"],
            "use_length": record["use_length"], "num_steps": record["num_steps"], "compile": record["compile"],
            "action_smoothing": record["action_smoothing"], "policy_rng": record["policy_rng"],
            "policy_rng_protocol": "process_global_torch_legacy", "expert_seed_policy": EXPERT_SEED_POLICY,
            "expert_cache_mode": "off", "trials": quick.TRIALS, "settings": list(quick.SETTINGS), "tasks": sorted(TASKS),
            "sim_workers": len(endpoints), "fixed_checkpoint": True,
            "remote_inference": {"server_host": record["host"], "server_endpoints": record["endpoints"],
                                 "client_endpoints": [f"{host}:{port}" for host, port in endpoints],
                                 "simulator_host": socket.gethostname()}}


def quick_plan(record, endpoints):
    return {"schema": quick.SCHEMA, "trials": quick.TRIALS, "settings": list(quick.SETTINGS),
            "checkpoint": record["checkpoint"], "config": "remote policy servers", "precision": record["precision"],
            "part": [1, 1], "part_tasks": sorted(TASKS), "sims_per_gpu": len(endpoints),
            "policy_rng": record["policy_rng"], "shards": [{"name": "s0", "tasks": sorted(TASKS)}]}


def prepare_output(output, record, endpoints):
    output = Path(output).resolve()
    shard = output / "shards" / "s0"
    shard.mkdir(parents=True, exist_ok=True)
    plan, evaluation = quick_plan(record, endpoints), evaluation_plan(record, endpoints)
    saved = output / "quick_plan.json"
    if saved.is_file():
        previous = json.loads(saved.read_text())
        old = json.loads((shard / "evaluation_plan.json").read_text())
        same_weights = (old.get("checkpoint_identity") or {}).get("sha256") == (evaluation["checkpoint_identity"] or {}).get("sha256")
        if previous.get("checkpoint") != plan["checkpoint"] or not same_weights or \
                any(old.get(key) != evaluation[key] for key in quick.PROTOCOL_KEYS):
            raise ValueError(f"{output} holds results of another checkpoint or protocol; choose a new --output.")
    quick.write_json(saved, plan)
    quick.write_json(shard / "evaluation_plan.json", evaluation)
    return output, plan


def simulator_environment(runtime, robotwin, vendor, conda_root):
    sim = Path(conda_root) / "envs/robotwin-sim"
    env = dict(os.environ, PYTHONNOUSERSITE="1", PYTHONUNBUFFERED="1", TOKENIZERS_PARALLELISM="false",
               SETUPTOOLS_SCM_PRETEND_VERSION="0.0.0", OMP_NUM_THREADS="1", MKL_NUM_THREADS="1",
               OPENBLAS_NUM_THREADS="1", NUMEXPR_NUM_THREADS="1")
    env.pop("LRVLA_RENDER_PCI", None)
    env["PYTHONPATH"] = os.pathsep.join(map(str, [runtime, vendor, robotwin, Path(robotwin) / "description/utils"]))
    env["PATH"] = os.pathsep.join(filter(None, [str(sim / "bin"), os.environ.get("PATH", "")]))
    env["LD_LIBRARY_PATH"] = os.pathsep.join(filter(None, [str(sim / "lib"), os.environ.get("LD_LIBRARY_PATH", "")]))
    return env, sim / "bin/python"


def client_command(python, client, task, setting, port, output):
    return [str(python), "-u", str(client), "--config", "policy/pi0/deploy_policy.yml", "--overrides",
            "--task_name", task, "--task_config", setting, "--ckpt_setting", "cotrain", "--train_config_name", "0",
            "--seed", "0", "--policy_name", "pi0", "--port", str(port), "--robo_name", "robotwin",
            "--instruction_type", INSTRUCTIONS[setting], "--eval_video_log", "False", "--output_dir", str(output)]


def archive(shard, paths):
    existing = [path for path in paths if path.exists()]
    if existing:
        target = shard / "previous_attempts" / f"{time.time_ns()}"
        for path in existing:
            destination = target / path.relative_to(shard)
            destination.parent.mkdir(parents=True, exist_ok=True)
            path.replace(destination)


def run_clients(output, plan, endpoints, *, client, python, env, cwd, limits, retries=1, progress_interval=300,
                endpoint_wait=600):
    """One simulator per endpoint at a time; unfinished task/setting pairs are pulled longest-first."""
    shard = Path(output) / "shards" / "s0"
    pairs = sorted(((setting, task) for setting in quick.SETTINGS for task in TASKS
                    if not evaluation_task_complete(shard, setting, task, quick.TRIALS)),
                   key=lambda pair: (-limits[pair[1]], pair[1], pair[0]))
    pending = queue.Queue()
    for pair in pairs:
        pending.put((pair, 0))
    processes, lock, stopped = {}, threading.Lock(), threading.Event()
    status_file = Path(output) / "status.json"
    attempts = json.loads(status_file.read_text()) if status_file.is_file() else []

    def endpoint_ready(host, port):
        deadline = time.monotonic() + endpoint_wait
        while not stopped.is_set():
            if port_open(host, port):
                return True
            if time.monotonic() > deadline:
                return False
            time.sleep(10)
        return False

    def worker(slot, host, port):
        while not stopped.is_set():
            try:
                (setting, task), attempt = pending.get_nowait()
            except queue.Empty:
                return
            if not endpoint_ready(host, port):
                pending.put(((setting, task), attempt))
                print(f"[remote-eval] slot {slot}: {host}:{port} unreachable; slot stopped", flush=True)
                return
            log = shard / setting / "eval_logs" / f"{task}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            status = log.with_suffix(".status.json")
            archive(shard, [log, status])
            started = time.monotonic()
            with log.open("w") as stream:
                with lock:
                    if stopped.is_set():
                        return
                    process = subprocess.Popen(client_command(python, client, task, setting, port, shard / setting / "eval_results"),
                                               cwd=cwd, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                    processes[process.pid] = process
                code = process.wait()
                with lock:
                    processes.pop(process.pid, None)
            status.write_text(json.dumps({"exit_code": code, "elapsed_seconds": time.monotonic() - started,
                                          "pid": process.pid, "slot": slot, "endpoint": f"{host}:{port}"}) + "\n")
            record = {"setting": setting, "task": task, "attempt": attempt + 1, "slot": slot, "exit_code": code,
                      "seconds": round(time.monotonic() - started, 1), "finished_at": now()}
            with lock:
                attempts.append(record)
                quick.write_json(status_file, attempts)
            print(f"[remote-eval] {setting}/{task} slot {slot}: exit {code}", flush=True)
            if code != 0 and not stopped.is_set() and attempt < retries:
                pending.put(((setting, task), attempt + 1))

    def terminate(signum, frame):
        stopped.set()
        with lock:
            for process in list(processes.values()):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    total = len(TASKS) * len(quick.SETTINGS) * quick.TRIALS
    total_work = sum(limits.values()) * len(quick.SETTINGS) * quick.TRIALS
    counts, start_work = quick.progress(output, plan, limits)
    started = time.monotonic()
    print(quick.progress_line(counts, total, 0, None), flush=True)
    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGINT, signal.SIGTERM)}
    try:
        with ThreadPoolExecutor(max_workers=len(endpoints)) as pool:
            futures = {pool.submit(worker, slot, host, port) for slot, (host, port) in enumerate(endpoints)}
            while futures:
                finished, futures = wait(futures, timeout=progress_interval, return_when=FIRST_COMPLETED)
                for future in finished:
                    future.result()
                counts, work = quick.progress(output, plan, limits)
                elapsed = time.monotonic() - started
                rate = (work - start_work) / elapsed if elapsed > 0 else 0
                print(quick.progress_line(counts, total, elapsed, (total_work - work) / rate if rate > 0 else None), flush=True)
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return not stopped.is_set()


def evaluate(manifest, endpoints, output, *, baseline=None, retries=1, progress_interval=300, robotwin=ROOT / "vendor/RoboTwin",
             vendor=ROOT / "vendor/lingbot-vla-v2", conda_root=None, client=None, python=None, preflight=True):
    record = load_manifest(manifest)
    if len(endpoints) > len(record["endpoints"]):
        raise ValueError("More client endpoints than policy servers in the manifest.")
    output, plan = prepare_output(output, record, endpoints)
    limits = quick.step_limits(robotwin)
    conda_root = conda_root or Path(os.environ.get("CONDA_ROOT") or Path.home() / "miniconda3")
    env, sim_python = simulator_environment(output / "runtime", robotwin, vendor, conda_root)
    if client is None:
        client = create_eval_runtime(vendor, robotwin, output / "runtime", quick.TRIALS, timing=True)
    if preflight:
        sys.path.insert(0, str(ROOT / "scripts"))
        from doctor import RENDER_CODE, isolated_probe
        report = isolated_probe(sim_python, RENDER_CODE, timeout=60, cwd=robotwin, env=env)
        quick.write_json(output / "render_preflight.json", report)
        if not report["ok"]:
            raise RuntimeError(f"Rendering preflight {report['status']}; read {output / 'render_preflight.json'}")
    print(f"[remote-eval] servers on {record['host']} ({record['checkpoint']}), {len(endpoints)} simulator(s), output {output}",
          flush=True)
    finished = run_clients(output, plan, endpoints, client=client, python=python or sim_python, env=env, cwd=robotwin,
                           limits=limits, retries=retries, progress_interval=progress_interval)
    summary = quick.collect([output], output, baseline)
    print(quick.render_summary(summary), flush=True)
    if not finished:
        return 130
    return 0 if summary["complete"] else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    server = commands.add_parser("serve", help="Start policy servers and write their manifest")
    server.add_argument("--checkpoint", type=Path, required=True)
    server.add_argument("--manifest", type=Path, required=True)
    server.add_argument("--ports", required=True, help="Comma-separated ports, one server each")
    server.add_argument("--gpus", required=True, help="Comma-separated CUDA device per port")
    server.add_argument("--qwen", type=Path, default=ROOT / "models/Qwen3-VL-4B-Instruct")
    client = commands.add_parser("evaluate", help="Run simulators against reachable endpoints")
    client.add_argument("--manifest", type=Path, required=True, help="Manifest written by `serve`")
    client.add_argument("--endpoints", type=parse_endpoints, required=True,
                        help="host:port per simulator, in manifest order, e.g. SSH-forwarded 127.0.0.1 ports")
    client.add_argument("--output", type=Path, required=True)
    client.add_argument("--baseline", type=Path)
    client.add_argument("--retries", type=int, default=1)
    client.add_argument("--progress-interval", type=int, default=300)
    args = parser.parse_args(argv)
    if args.command == "serve":
        ports = [int(port) for port in args.ports.split(",")]
        gpus = [gpu.strip() for gpu in args.gpus.split(",")]
        return serve(args.checkpoint, args.manifest, ports, gpus, qwen=args.qwen)
    return evaluate(args.manifest, args.endpoints, args.output, baseline=args.baseline, retries=args.retries,
                    progress_interval=args.progress_interval)
