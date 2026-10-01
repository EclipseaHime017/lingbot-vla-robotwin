#!/usr/bin/env python3
"""Check explicit Linux tool paths; leave PATH and other variables unchanged."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from lrvla.wsl_tools import linux_executable

if __name__ == "__main__":
    for item in sys.argv[1:]:
        name, separator, candidate = item.partition("=")
        path = linux_executable(name, preferred=candidate if separator else None)
        print(f"{name}: {path}")
