#!/usr/bin/env python3
"""Verify pinned SHA256 source metadata in an unpacked submission vendor tree."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.vendor_sources import verify_snapshot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repository", type=Path)
    parser.add_argument("expected_revision")
    args = parser.parse_args()
    try:
        metadata = verify_snapshot(args.repository, args.expected_revision)
    except (ValueError, OSError) as error:
        parser.exit(1, f"Vendor source verification failed: {error}\n")
    print(json.dumps({"repository": str(args.repository.resolve()), "revision": metadata["revision"],
                      "verified_files": len(metadata["files"])}))


if __name__ == "__main__":
    main()
