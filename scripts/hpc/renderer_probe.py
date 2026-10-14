#!/usr/bin/env python3
"""Test actual GPU offscreen camera rendering before a RoboTwin evaluation."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts"))
from doctor import PHYSICS_CODE, RENDER_CODE, isolated_probe, vulkan_icds  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not os.environ.get("PBS_JOBID"):
        parser.error("Run the native GPU probe in a PBS allocation")
    conda = Path(os.environ.get("CONDA_ROOT", str(Path.home() / "miniconda3")))
    python = conda / "envs/robotwin-sim/bin/python"
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
        capture_output=True, text=True, timeout=10, check=False,
    )
    env = {"LD_LIBRARY_PATH": str(python.parent.parent / "lib") + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")}
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "job_id": os.environ["PBS_JOBID"],
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": {"returncode": gpu.returncode, "stdout": gpu.stdout, "stderr": gpu.stderr},
        "vulkan_icds": vulkan_icds(),
        "nvidia_graphics_libraries": [
            str(path)
            for directory in (Path("/usr/lib64"), Path("/usr/lib/x86_64-linux-gnu"))
            for pattern in ("libGLX_nvidia.so*", "libEGL_nvidia.so*", "libnvidia-glvkspirv.so*")
            for path in directory.glob(pattern)
        ],
        "physics": isolated_probe(python, PHYSICS_CODE, timeout=30, cwd=ROOT, env=env),
        "camera_rendering": isolated_probe(python, RENDER_CODE, timeout=45, cwd=ROOT, env=env),
        "purpose": "Prerequisite for visual closed-loop RoboTwin evaluation; not a policy success-rate test",
    }
    report["ok"] = report["physics"]["ok"] and report["camera_rendering"]["ok"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2), flush=True)
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
