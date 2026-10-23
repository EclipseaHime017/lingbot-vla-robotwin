#!/usr/bin/env python3
"""Fixed 50 tasks x 2 settings x 10 scenes screening; see lrvla.quick_evaluation."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lrvla.quick_evaluation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
