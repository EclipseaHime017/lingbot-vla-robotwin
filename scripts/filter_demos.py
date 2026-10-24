#!/usr/bin/env python3
"""Screen clean demonstrations and build a filtered manifest; see lrvla.demo_filter.

  scan:     official clean archives -> reviewable exclusion list (configs/demo_filter.json)
  manifest: clean manifest + exclusion list -> filtered manifest and its normalization statistics
"""
import argparse
from concurrent.futures import ProcessPoolExecutor
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import TASKS, file_sha256  # noqa: E402
from lrvla import demo_filter  # noqa: E402


def scan_task(arguments):
    task, archive = arguments
    return demo_filter.scan_archive(archive, task)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    scan = commands.add_parser("scan")
    scan.add_argument("--archives", type=Path, default=ROOT / "data/raw_archives/dataset")
    scan.add_argument("--output", type=Path, default=ROOT / "configs/demo_filter.json")
    scan.add_argument("--workers", type=int, default=8)
    build = commands.add_parser("manifest")
    build.add_argument("--manifest", type=Path, required=True, help="Audited clean manifest covering all 50 episodes")
    build.add_argument("--filter", type=Path, default=ROOT / "configs/demo_filter.json")
    build.add_argument("--output", type=Path, required=True)
    build.add_argument("--norm-output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "scan":
        archives = {task: args.archives / task / "aloha-agilex_clean_50.zip" for task in TASKS}
        official = {item["rfilename"]: item["lfs"]["sha256"]
                    for item in json.loads((ROOT / "configs/raw_clean_hf_metadata.json").read_text())["files"]}
        digests = {}
        for task, archive in archives.items():
            digests[task] = file_sha256(archive)
            if digests[task] != official[f"dataset/{task}/aloha-agilex_clean_50.zip"]:
                raise ValueError(f"{archive} differs from the official clean archive")
        with ProcessPoolExecutor(args.workers) as pool:
            rows = [row for part in pool.map(scan_task, archives.items()) for row in part]
        record = demo_filter.exclusion_record(rows, digests)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(record, indent=2) + "\n")
        print(json.dumps({key: record[key] for key in ("filter_version", "episodes_scanned", "episodes_excluded")}))
        return 0
    from lrvla.data import load_manifest
    manifest = load_manifest(args.manifest)
    if not manifest.get("complete_50_tasks"):
        raise ValueError("Filter the complete 50-task manifest.")
    filtered = demo_filter.filtered_manifest(manifest, json.loads(args.filter.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(filtered, indent=2) + "\n")
    result = demo_filter.compute_norm_for_manifest(args.output, args.norm_output)
    print(json.dumps({"manifest": str(args.output), "episodes": filtered["total_episodes"], "frames": result["count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
