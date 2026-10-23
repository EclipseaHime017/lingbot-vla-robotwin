#!/usr/bin/env python3
"""Warm-start BC with whitelisted augmentations; scripts/train.py stays the original entry."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lrvla.warm_start import main

if __name__ == "__main__":
    main()
