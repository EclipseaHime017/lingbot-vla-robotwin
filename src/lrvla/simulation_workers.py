"""Bounded independent simulator clients owned by one policy-server process."""
from __future__ import annotations

from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import queue
import subprocess
import threading
import time


def run_clients(commands, *, run, cwd, environment, workers, server, archive, stop, environments=None):
    """Run at most one simulator per worker slot.

    `environments` optionally gives each slot its own environment, e.g. a
    simulator GPU other than the policy server's. A slot is held for the whole
    lifetime of its process, so per-GPU simulator concurrency stays bounded.
    """
    if environments is not None and len(environments) != workers:
        raise ValueError("Provide exactly one simulator environment per worker slot.")
    pool = ThreadPoolExecutor(max_workers=workers)
    processes, lock, stopping = {}, threading.Lock(), threading.Event()
    slots = queue.Queue()
    for slot in range(workers):
        slots.put(slot)

    def execute(item):
        if stopping.is_set():
            return
        logfile = run / item["setting"] / "eval_logs" / f"{item['task']}.log"
        logfile.parent.mkdir(parents=True, exist_ok=True)
        status = logfile.with_suffix(".status.json")
        archive(run, [logfile, status])
        slot = slots.get()
        try:
            slot_environment = environment if environments is None else environments[slot]
            started = time.monotonic()
            with logfile.open("w") as output:
                with lock:
                    if stopping.is_set():
                        return
                    process = subprocess.Popen(item["command"], cwd=cwd, env=slot_environment,
                                               stdout=output, stderr=subprocess.STDOUT, start_new_session=True)
                    processes[process.pid] = process
                code = process.wait()
            status.write_text(json.dumps({"exit_code": code, "elapsed_seconds": time.monotonic() - started,
                                         "pid": process.pid, "slot": slot,
                                         "gpu": slot_environment.get("CUDA_VISIBLE_DEVICES")}) + "\n")
            with lock:
                processes.pop(process.pid, None)
        finally:
            slots.put(slot)
        print(f"{item['setting']}/{item['task']}: exit={code}", flush=True)

    try:
        pending = {pool.submit(execute, item) for item in commands}
        while pending:
            finished, pending = wait(pending, timeout=1, return_when=FIRST_COMPLETED)
            for future in finished:
                future.result()
            if pending and server.poll() is not None:
                raise RuntimeError("Shared policy server exited while simulator clients were active.")
    finally:
        stopping.set()
        with lock:
            active = list(processes.values())
        # Stop all clients before waiting for executor teardown, including on
        # SIGTERM/KeyboardInterrupt; blocked websocket receives must not hang it.
        for process in active:
            stop(process)
        pool.shutdown(wait=True, cancel_futures=True)
