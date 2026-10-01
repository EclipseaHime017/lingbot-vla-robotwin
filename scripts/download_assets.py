#!/usr/bin/env python3
"""Resumable, revision-pinned public Hugging Face downloads with SHA-256 checks.

Only official RoboTwin aloha-agilex clean archives are selected. The public
robbyant EEF LeRobot dataset has 16-D actions and is unsuitable for V2 joint14.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import time
from urllib.parse import quote
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
BASE_REPO = "robbyant/lingbot-vla-v2-6b"
BASE_REV = "11c703bf6a5c1f45b3b69168482da11fdbba53d7"
QWEN_REPO = "Qwen/Qwen3-VL-4B-Instruct"
QWEN_REV = "ebb281ec70b05090aa6165b016eac8ec08e71b17"
DATA_REPO = "TianxingChen/RoboTwin2.0"
DATA_REV = "3dc3b798668feb99ac61cc9086d84cbcc3d79186"


def request_json(url: str):
    with urlopen(Request(url, headers={"User-Agent": "lingbot-vla-robotwin/1.0"}), timeout=90) as response:
        return json.load(response)


def hf_metadata(repo: str, kind: str, revision: str) -> dict:
    result = request_json(f"https://huggingface.co/api/{kind}s/{repo}/revision/{revision}?blobs=true")
    if result.get("sha") != revision:
        raise RuntimeError(f"Requested immutable revision {revision}, API returned {result.get('sha')}")
    return result


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download_file(url: str, destination: Path, size: int, expected_hash: str | None, retries: int = 8) -> dict:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size:
        actual = sha256(destination) if expected_hash else None
        if not expected_hash or actual == expected_hash:
            return {"path": str(destination), "size": size, "sha256": actual, "status": "verified_existing"}
    partial = destination.with_name(destination.name + ".part")
    for attempt in range(retries):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset > size:
            partial.unlink()
            offset = 0
        try:
            if offset < size:
                headers = {"User-Agent": "lingbot-vla-robotwin/1.0"}
                if offset:
                    headers["Range"] = f"bytes={offset}-"
                with urlopen(Request(url, headers=headers), timeout=120) as response:
                    if offset and response.status != 206:
                        offset = 0
                    elif offset:
                        match = re.match(r"bytes (\d+)-", response.headers.get("Content-Range", ""))
                        if not match or int(match.group(1)) != offset:
                            raise RuntimeError("Server returned an unexpected Content-Range")
                    with partial.open("ab" if offset else "wb") as stream:
                        shutil.copyfileobj(response, stream, length=8 * 1024 * 1024)
            if partial.stat().st_size != size:
                raise RuntimeError(f"Incomplete download {partial}: {partial.stat().st_size}/{size} bytes")
            actual = sha256(partial)
            if expected_hash and actual != expected_hash:
                partial.unlink()
                raise RuntimeError(f"SHA-256 mismatch for {destination.name}")
            partial.replace(destination)
            print(f"verified {destination.name}: {size:,} bytes", flush=True)
            return {"path": str(destination), "size": size, "sha256": actual, "status": "downloaded"}
        except Exception as error:
            if attempt + 1 == retries:
                raise
            print(f"retry {attempt + 1}/{retries} {destination.name}: {error}", flush=True)
            time.sleep(min(2 ** attempt, 30))
    raise AssertionError("unreachable")


def plan_assets(asset: str, tasks: list[str] | None = None) -> list[dict]:
    selections = []
    if asset in ("base", "all"):
        metadata = hf_metadata(BASE_REPO, "model", BASE_REV)
        files = [item for item in metadata["siblings"] if not item["rfilename"].startswith(("assets/", "depth/", "dino_video/"))]
        selections.append({"repo": BASE_REPO, "revision": BASE_REV, "kind": "model", "destination": ROOT / "models/lingbot-vla-v2-6b", "files": files})
    if asset in ("qwen", "all"):
        metadata = hf_metadata(QWEN_REPO, "model", QWEN_REV)
        files = [item for item in metadata["siblings"] if not item["rfilename"].endswith(".safetensors") and item["rfilename"] != "model.safetensors.index.json"]
        selections.append({"repo": QWEN_REPO, "revision": QWEN_REV, "kind": "model", "destination": ROOT / "models/Qwen3-VL-4B-Instruct", "files": files})
    if asset in ("teachers",):
        metadata = hf_metadata(BASE_REPO, "model", BASE_REV)
        files = [item for item in metadata["siblings"] if item["rfilename"].startswith(("depth/", "dino_video/"))]
        selections.append({"repo": BASE_REPO, "revision": BASE_REV, "kind": "model", "destination": ROOT / "models/lingbot-vla-v2-6b", "files": files})
    if asset in ("clean", "all"):
        metadata = hf_metadata(DATA_REPO, "dataset", DATA_REV)
        files = [item for item in metadata["siblings"] if re.fullmatch(r"dataset/[^/]+/aloha-agilex_clean_50\.zip", item["rfilename"])]
        if len(files) != 50:
            raise RuntimeError(f"Expected 50 official aloha-agilex clean archives, found {len(files)}")
        if tasks:
            files = [item for item in files if item["rfilename"].split("/")[1] in tasks]
            if len(files) != len(set(tasks)):
                raise ValueError("One or more requested tasks do not exist in the official clean dataset")
        selections.append({"repo": DATA_REPO, "revision": DATA_REV, "kind": "dataset", "destination": ROOT / "data/raw_archives", "files": files})
    if asset == "sim":
        metadata = hf_metadata(DATA_REPO, "dataset", DATA_REV)
        files = [item for item in metadata["siblings"] if item["rfilename"] in ("objects.zip", "embodiments.zip", "background_texture.zip")]
        if len(files) != 3:
            raise RuntimeError("Expected the three official RoboTwin simulator asset archives")
        selections.append({"repo": DATA_REPO, "revision": DATA_REV, "kind": "dataset", "destination": ROOT / "data/sim_archives", "files": files})
    return selections


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset", choices=["base", "qwen", "teachers", "clean", "sim", "all"], default="all")
    parser.add_argument("--tasks", help="Comma-separated clean task subset for smoke preparation")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--plan", action="store_true", help="Inspect sizes and pin provenance without downloading")
    args = parser.parse_args()
    jobs = plan_assets(args.asset, args.tasks.split(",") if args.tasks else None)
    report = {"asset": args.asset, "download_complete": False, "sources": []}
    total = 0
    for job in jobs:
        count_bytes = sum(item["size"] for item in job["files"])
        total += count_bytes
        print(f"{job['repo']}@{job['revision']}: {len(job['files'])} files, {count_bytes / 1e9:.3f} GB", flush=True)
        report["sources"].append({**{k: str(v) if isinstance(v, Path) else v for k, v in job.items() if k != "files"}, "files": job["files"]})
    report_path = ROOT / "artifacts" / f"download_{args.asset}{'_subset' if args.tasks else ''}.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    if args.plan:
        return
    if shutil.disk_usage(ROOT).free < total * 1.05:
        raise RuntimeError(f"Insufficient disk space for {total / 1e9:.1f} GB asset plan")
    for job, source in zip(jobs, report["sources"]):
        def run(item):
            filename = item["rfilename"]
            if filename.startswith("/") or ".." in Path(filename).parts:
                raise ValueError("Unsafe repository path")
            prefix = "datasets/" if job["kind"] == "dataset" else ""
            url = f"https://huggingface.co/{prefix}{job['repo']}/resolve/{job['revision']}/{quote(filename)}"
            digest = item.get("lfs", {}).get("sha256")
            return download_file(url, job["destination"] / filename, item["size"], digest)
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            source["downloads"] = list(pool.map(run, job["files"]))
        report_path.write_text(json.dumps(report, indent=2) + "\n")
    report["download_complete"] = True
    report_path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"Download provenance: {report_path}")


if __name__ == "__main__":
    main()
