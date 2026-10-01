#!/usr/bin/env python3
"""Unpack hash-verified public simulation assets without replacing tracked files."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import zipfile
import zlib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import file_sha256


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robotwin", type=Path, default=ROOT / "vendor/RoboTwin")
    parser.add_argument("--archives", type=Path, default=ROOT / "data/sim_archives")
    args = parser.parse_args()
    plan = json.loads((ROOT / "artifacts/download_sim.json").read_text())
    destination = (args.robotwin / "assets").resolve()
    for item in plan["sources"][0]["files"]:
        archive = args.archives / item["rfilename"]
        if file_sha256(archive) != item["lfs"]["sha256"]:
            raise ValueError(f"Simulation asset archive hash mismatch: {archive}")
        with zipfile.ZipFile(archive) as bundle:
            size = sum(member.file_size for member in bundle.infolist())
            if shutil.disk_usage(ROOT).free < size:
                raise RuntimeError(f"Insufficient free disk space to extract {archive.name}: {size / 1e9:.2f} GB")
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if not target.is_relative_to(destination) or (member.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("Unsafe simulation archive member")
                if target.exists() and not member.is_dir():
                    # Resume completed members; never replace a differing file.
                    if target.stat().st_size != member.file_size:
                        raise ValueError(f"Asset extraction would replace differing existing file: {target}")
                    checksum = 0
                    with target.open("rb") as stream:
                        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                            checksum = zlib.crc32(block, checksum)
                    if checksum != member.CRC:
                        raise ValueError(f"Existing simulation asset differs from the official archive: {target}")
                    continue
                bundle.extract(member, destination)
        print(f"Extracted {archive.name} into {destination}", flush=True)
    print("Run robotwin-sim Python vendor/RoboTwin/script/update_embodiment_config_path.py from the RoboTwin directory to regenerate absolute URDF paths.")


if __name__ == "__main__":
    main()
