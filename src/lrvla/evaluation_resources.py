"""Periodic GPU samples for evaluating simulator concurrency on real hardware."""
from __future__ import annotations

import json
import subprocess
import threading
import time

from .wsl_tools import linux_executable


class ResourceSampler:
    def __init__(self, path, gpu):
        self.path, self.gpu = path, gpu
        self.finished = threading.Event()
        self.thread = threading.Thread(target=self.sample, daemon=True)

    def start(self):
        self.thread.start()

    def sample(self):
        fields = "uuid,utilization.gpu,memory.used,memory.total,power.draw"
        while not self.finished.is_set():
            row = {"timestamp": time.time(), "fields": fields}
            try:
                result = subprocess.run([str(linux_executable("nvidia-smi")), "-i", self.gpu,
                                         "--query-gpu=" + fields, "--format=csv,noheader,nounits"],
                                        capture_output=True, text=True, timeout=5, check=True)
                row["values"] = result.stdout.strip()
            except Exception as error:
                row["error"] = str(error)
            with self.path.open("a") as stream:
                stream.write(json.dumps(row) + "\n")
            self.finished.wait(5)

    def close(self):
        self.finished.set()
        self.thread.join(timeout=7)
