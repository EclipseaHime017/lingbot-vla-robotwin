#!/usr/bin/env python3
"""Run one independent model/simulator worker per assigned GPU."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.parallel_evaluation import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())
