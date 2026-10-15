"""Independent per-GPU policy/simulator workers sharing one fixed protocol."""
from __future__ import annotations

import argparse
import ast
from concurrent.futures import ThreadPoolExecutor
import fcntl
import importlib.util
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

from .data import parse_tasks
from .render_device import normalized_pci
from .wsl_tools import linux_executable

ROOT = Path(__file__).resolve().parents[2]


def assigned_devices(selectors=None, *, environment=None):
    environment = os.environ if environment is None else environment
    assigned = environment.get("CUDA_VISIBLE_DEVICES", "")
    requested = selectors or assigned
    if not requested:
        raise ValueError("Specify --gpus or CUDA_VISIBLE_DEVICES.")
    executable = linux_executable("nvidia-smi")

    def resolve(value):
        result = subprocess.run([str(executable), "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader", "-i", value],
                                capture_output=True, text=True, check=True)
        rows = result.stdout.strip().splitlines()
        if len(rows) != 1:
            raise ValueError(f"GPU selector must identify one physical device: {value}")
        uuid, pci = (part.strip() for part in rows[0].split(","))
        return {"uuid": uuid, "pci": normalized_pci(pci)}

    devices = [resolve(value.strip()) for value in requested.split(",")]
    if len({device["uuid"] for device in devices}) != len(devices):
        raise ValueError("Duplicate GPU assignment.")
    if assigned and selectors:
        allowed = {resolve(value.strip())["uuid"] for value in assigned.split(",")}
        if not {device["uuid"] for device in devices} <= allowed:
            raise ValueError("Requested GPU is outside CUDA_VISIBLE_DEVICES.")
    return devices


def select_units(workflow, tasks):
    selected = set(parse_tasks(",".join(workflow["tasks"]) if tasks == "all" else tasks))
    units, covered, outputs = [], set(), set()
    if workflow.get("submission_disabled"):
        raise ValueError("This evaluation workflow is disabled.")
    for shard in workflow["shards"]:
        members = set(shard["tasks"])
        if not members & selected:
            continue
        if not members <= selected:
            raise ValueError("Select whole shards when reusing their existing evaluation plan.")
        output = str(Path(shard["output"]).resolve())
        if covered & members or output in outputs:
            raise ValueError("Overlapping tasks or output directories in parallel evaluation.")
        covered.update(members)
        outputs.add(output)
        units.append(shard)
    if covered != selected:
        raise ValueError("Requested tasks are missing from the evaluation workflow.")
    return units


def evaluation_arguments(workflow, shard, device, port):
    if workflow["inference_backend"] not in {"native", "streamed", "streamed_sdpa_reference_moe"}:
        raise ValueError("Unknown inference backend.")
    settings = workflow["settings"]
    if set(settings) == {"demo_clean", "demo_randomized"}:
        setting = "both"
    elif len(settings) == 1 and settings[0] in {"demo_clean", "demo_randomized"}:
        setting = settings[0]
    else:
        raise ValueError("Unexpected evaluation settings.")
    arguments = ["--checkpoint", workflow["checkpoint"], "--tasks", ",".join(shard["tasks"]),
                 "--setting", setting, "--trials", str(workflow["trials"]),
                 "--precision", workflow["precision"], "--use-length", str(workflow["use_length"]),
                 "--output", shard["output"], "--skip-completed", "--gpu", device["uuid"], "--port", str(port)]
    if workflow["inference_backend"] == "native":
        arguments.append("--native-server")
    elif "num_steps" in workflow:
        arguments.extend(["--backend", "streamed"])
    if "num_steps" in workflow:
        arguments.extend(["--num-steps", str(workflow["num_steps"])])
        arguments.append("--compile" if workflow["compile"] else "--no-compile")
        if workflow.get("action_smoothing") != "none":
            raise ValueError("Unsupported action smoothing profile.")
    if "sim_workers" in workflow:
        arguments.extend(["--sim-workers", str(workflow["sim_workers"]), "--policy-rng", workflow["policy_rng"]])
    if workflow.get("expert_cache_mode", "off") != "off":
        arguments.extend(["--expert-cache-mode", workflow["expert_cache_mode"], "--expert-cache", workflow["expert_cache"]])
    # Legacy v1 evaluate.py has fixed 10 denoising steps and compile=False.
    # Do not silently pass new v2 CLI fields into a running v1 workflow.
    return arguments


def binding_bootstrap():
    return (f"import sys\nsys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "from lrvla.render_device import bind_render_device\nbind_render_device()\n")


def inject_binding(source):
    tree = ast.parse(source)
    line = 0
    for node in tree.body:
        if (isinstance(node, ast.Expr) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str)) or (
                isinstance(node, ast.ImportFrom) and node.module == "__future__"):
            line = node.end_lineno
        else:
            break
    lines = source.splitlines(keepends=True)
    lines.insert(line, binding_bootstrap())
    result = "".join(lines)
    ast.parse(result)
    return result


def worker_entry(script, arguments):
    """Wrap the selected evaluator without replacing its protocol modules."""
    script = Path(script).resolve()
    sys.path.insert(0, str(script.parent))
    package = script.parent.parent / "src/lrvla"
    if package.is_dir():
        # lrvla is already imported by this entry point. Changing sys.path alone
        # would leave its module search path pointing at the launcher's release.
        import lrvla
        loaded = sys.modules.get("lrvla.evaluation")
        if loaded is not None and Path(loaded.__file__).resolve().parent != package.resolve():
            raise RuntimeError("A different evaluator release is already imported.")
        lrvla.__path__ = [str(package), *[path for path in lrvla.__path__ if path != str(package)]]
        sys.path.insert(0, str(package.parent))
    spec = importlib.util.spec_from_file_location("lrvla_worker_evaluator", script)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    if evaluator.EXPERT_SEED_POLICY != os.environ["LRVLA_EXPECT_EXPERT_SEED_POLICY"]:
        raise ValueError("Evaluator protocol differs from this fixed workflow.")
    if not getattr(evaluator, "RENDER_BINDING", False):
        # Older fixed-protocol evaluators need additive runtime hooks.
        original_create = evaluator.create_eval_runtime

        def create_runtime(*args, **kwargs):
            path = original_create(*args, **kwargs)
            path.write_text(inject_binding(path.read_text()))
            return path

        evaluator.create_eval_runtime = create_runtime
        import doctor
        doctor.RENDER_CODE = inject_binding(doctor.RENDER_CODE)
    sys.argv = [str(script), *arguments]

    def terminate(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, terminate)
    return evaluator.main()


def available_port():
    with socket.socket() as connection:
        connection.bind(("127.0.0.1", 0))
        return connection.getsockname()[1]


def run_parallel(workflow, units, devices, *, evaluate_script, log_dir, python=sys.executable):
    log_dir = Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    pending = queue.Queue()
    for unit in units:
        pending.put(unit)
    results, processes, lock, ports = [], {}, threading.Lock(), set()
    stopped = threading.Event()

    def worker(device):
        while not stopped.is_set():
            try:
                unit = pending.get_nowait()
            except queue.Empty:
                return
            output = Path(unit["output"])
            output.mkdir(parents=True, exist_ok=True)
            record = {"tasks": unit["tasks"], "gpu": device, "output": str(output), "started_at": time.time()}
            with (output / ".parallel-eval.lock").open("a") as ownership:
                try:
                    fcntl.flock(ownership, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    with lock:
                        for _ in range(100):
                            port = available_port()
                            if port not in ports:
                                ports.add(port)
                                break
                        else:
                            raise RuntimeError("Could not allocate a distinct inference port.")
                    command = [str(python), str(ROOT / "scripts/evaluate_parallel.py"), "--worker-entry",
                               str(evaluate_script), *evaluation_arguments(workflow, unit, device, port)]
                    env = dict(os.environ, CUDA_VISIBLE_DEVICES=device["uuid"], LRVLA_RENDER_PCI=device["pci"],
                               LRVLA_EXPECT_EXPERT_SEED_POLICY=workflow["expert_seed_policy"],
                               PYTHONUNBUFFERED="1", PYTHONNOUSERSITE="1")
                    logfile = log_dir / ("-".join(unit["tasks"]) + ".log")
                    with logfile.open("a") as output_log:
                        with lock:
                            if stopped.is_set():
                                return
                            process = subprocess.Popen(command, env=env, stdout=output_log,
                                                       stderr=subprocess.STDOUT, start_new_session=True)
                            processes[process.pid] = process
                        record.update(exit_code=process.wait(), port=port, log=str(logfile))
                        with lock:
                            processes.pop(process.pid, None)
                except Exception as error:
                    record.update(exit_code=1, error=str(error))
            record["finished_at"] = time.time()
            with lock:
                results.append(record)
                temporary = log_dir / "parallel_status.json.tmp"
                temporary.write_text(json.dumps(results, indent=2) + "\n")
                temporary.replace(log_dir / "parallel_status.json")
            print(json.dumps(record), flush=True)

    def terminate(signum, frame):
        stopped.set()
        with lock:
            for process in list(processes.values()):
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass

    previous = {sig: signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
    try:
        with ThreadPoolExecutor(max_workers=len(devices)) as pool:
            futures = [pool.submit(worker, device) for device in devices]
            for future in futures:
                future.result()
    finally:
        for sig, handler in previous.items():
            signal.signal(sig, handler)
    return 0 if len(results) == len(units) and all(row["exit_code"] == 0 for row in results) else 1


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--worker-entry"]:
        return worker_entry(argv[1], argv[2:])
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--workflow", type=Path)
    source.add_argument("--config", type=Path, help="Prepare grouped tasks from a parallel evaluation YAML")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--sim-workers", type=int, help="Override YAML simulators per GPU")
    parser.add_argument("--expert-cache", type=Path)
    parser.add_argument("--expert-cache-mode", choices=["off", "record", "replay"])
    parser.add_argument("--prepare-only", action="store_true", help="Write the workflow without loading CUDA or rendering")
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--gpus", help="Allocated GPU UUIDs or nvidia-smi indices; defaults to CUDA_VISIBLE_DEVICES")
    parser.add_argument("--evaluate-script", type=Path, default=ROOT / "scripts/evaluate.py")
    parser.add_argument("--log-dir", type=Path)
    args = parser.parse_args(argv)
    if args.config:
        if not args.checkpoint or not args.output:
            parser.error("--config requires --checkpoint and --output")
        from .evaluation_workflow import prepare_workflow
        selectors = args.gpus or os.environ.get("CUDA_VISIBLE_DEVICES", "0")
        args.workflow = prepare_workflow(args.config, args.checkpoint, args.output, len(selectors.split(",")),
                                         overrides={"sim_workers": args.sim_workers,
                                                    "expert_cache": str(args.expert_cache) if args.expert_cache else None,
                                                    "expert_cache_mode": args.expert_cache_mode})
    elif any(value is not None for value in (args.checkpoint, args.output, args.sim_workers,
                                             args.expert_cache, args.expert_cache_mode)):
        parser.error("Set profile overrides when preparing --config; a saved --workflow is fixed")
    if args.prepare_only:
        print(f"Prepared workflow: {args.workflow}")
        return 0
    workflow = json.loads(args.workflow.read_text())
    args.log_dir = args.log_dir or args.workflow.parent / "workers"
    units, devices = select_units(workflow, args.tasks), assigned_devices(args.gpus)
    print(json.dumps({"devices": devices, "tasks": [unit["tasks"] for unit in units],
                      "evaluator": str(args.evaluate_script.resolve())}), flush=True)
    code = run_parallel(workflow, units, devices, evaluate_script=args.evaluate_script.resolve(), log_dir=args.log_dir)
    if workflow.get("schema") == "lrvla-parallel-evaluation-v1" and args.tasks == "all":
        from .evaluation_workflow import aggregate_workflow
        report = aggregate_workflow(workflow)
        return code or (0 if report["complete"] else 1)
    return code
