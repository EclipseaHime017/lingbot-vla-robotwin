#!/usr/bin/env python3
"""Policy servers and RoboTwin simulators on different machines; see lrvla.remote_evaluation.

  serve:    python scripts/evaluate_remote.py serve --checkpoint .../hf_ckpt --manifest servers.json --ports 9601,9602 --gpus 0,1
  evaluate: python scripts/evaluate_remote.py evaluate --manifest servers.json --endpoints 127.0.0.1:9601,127.0.0.1:9602 --output DIR
"""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lrvla.remote_evaluation import main  # noqa: E402

if __name__ == "__main__":
    raise SystemExit(main())
