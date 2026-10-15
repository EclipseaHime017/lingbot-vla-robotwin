#!/usr/bin/env python3
"""Build a seen-only sidecar from the same audited raw clean demonstrations."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.data import load_manifest  # noqa: E402
from lrvla.training_instructions import load_seen_pool  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/clean_manifest.json")
    parser.add_argument("--raw-root", type=Path, default=ROOT / "data/raw")
    parser.add_argument("--output", type=Path, default=ROOT / "data/clean_seen_instructions.json")
    args = parser.parse_args()
    manifest = load_manifest(args.manifest)
    result = {"schema": "clean_episode_seen_v1", "clean_only": True, "tasks": {}}
    for entry in manifest["datasets"]:
        candidates = list((args.raw_root / entry["task"]).rglob("source_provenance.json"))
        roots = [p.parent for p in candidates if all(json.loads(p.read_text()).get(key) == entry["source"].get(key)
                 for key in ("sha256", "setting", "revision", "archive", "repo_id", "embodiment"))]
        if len(roots) != 1:
            raise ValueError(f"Expected exactly one matching raw clean source: {entry['task']}")
        episodes = {}
        for episode in entry["episode_ids"]:
            # The unseen field is never selected or used.
            episodes[str(episode)] = json.loads((roots[0] / "instructions" / f"episode{episode}.json").read_text())["seen"]
        result["tasks"][entry["task"]] = {"source_sha256": entry["source"]["sha256"], "episodes": episodes}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    _, digest = load_seen_pool(args.output, manifest)
    print(json.dumps({"path": str(args.output), "content_sha256": digest, "tasks": len(result["tasks"])}))


if __name__ == "__main__":
    main()
