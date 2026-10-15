"""Small client-side stage timers, copied into the isolated simulator runtime."""
from __future__ import annotations

from contextlib import contextmanager
import json
import time


class EpisodeMetrics:
    def __init__(self, task, setting, episode, seed):
        self.data = {"task": task, "setting": setting, "episode": episode, "start_seed": seed,
                     "inference_calls": 0, "server_infer_seconds": 0.0}
        self.started = time.perf_counter()
        self.phase = None
        self.phase_started = self.started

    def phase_start(self, phase):
        now = time.perf_counter()
        if self.phase:
            self.data[self.phase] = self.data.get(self.phase, 0.0) + now - self.phase_started
        self.phase, self.phase_started = phase, now

    @contextmanager
    def measure(self, stage):
        started = time.perf_counter()
        try:
            yield
        finally:
            self.data[stage] = self.data.get(stage, 0.0) + time.perf_counter() - started

    def inference(self, timing):
        self.data["inference_calls"] += 1
        if isinstance(timing, dict):
            self.data["server_infer_seconds"] += float(timing.get("infer_ms", 0)) / 1000

    def finish(self, *, seed, success, initialization_failed=False):
        self.phase_start(None)
        self.data.update(policy_seed=seed, success=success, initialization_failed=initialization_failed,
                         total_seconds=time.perf_counter() - self.started)
        self.data["rpc_wait_and_transport_seconds"] = max(0.0, self.data.get("rpc_seconds", 0.0) - self.data["server_infer_seconds"])
        print("LRVLA_EPISODE_METRICS=" + json.dumps(self.data), flush=True)
