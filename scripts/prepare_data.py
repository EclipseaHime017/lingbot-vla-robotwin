#!/usr/bin/env python3
"""Convert and audit the 50 official clean demonstrations for each RoboTwin task."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import SETTING, parse_tasks, extract_clean_archive, convert_raw_task, prepare_manifest, compute_clean_norm


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tasks", default="all")
    parser.add_argument("--archives", type=Path, default=ROOT / "data/raw_archives/dataset")
    parser.add_argument("--data-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/lerobot")
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/clean_manifest.json")
    parser.add_argument("--norm-output", type=Path, default=ROOT / "data/clean_norm_stats.json")
    parser.add_argument("--val-episodes", type=int, default=5)
    parser.add_argument("--allow-subset", action="store_true")
    parser.add_argument("--audit-only", action="store_true")
    parser.add_argument("--skip-norm", action="store_true")
    args = parser.parse_args()
    tasks = parse_tasks(args.tasks)
    if len(tasks) != 50 and not args.allow_subset:
        parser.error("A task subset requires --allow-subset and is incomplete for submission")
    if not args.audit_only:
        for task in tasks:
            raw = extract_clean_archive(args.archives / task / f"{SETTING}.zip", task, args.data_root)
            convert_raw_task(raw, args.output / f"{task}-{SETTING}-50", task)
    manifest = prepare_manifest(args.output, tasks, args.manifest, args.allow_subset)
    if not args.skip_norm:
        compute_clean_norm(args.manifest, args.norm_output, args.val_episodes)
    print(f"{manifest['status']}: {len(tasks)} tasks / {manifest['total_episodes']} clean episodes")
    print(f"Manifest: {args.manifest.resolve()}")


if __name__ == "__main__":
    main()
