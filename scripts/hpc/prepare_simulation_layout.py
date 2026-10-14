#!/usr/bin/env python3
"""Keep the complete pinned RoboTwin tree, including regular assets, in scratch."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import stat
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lrvla.evaluation import ROBOTWIN_REVISION  # noqa: E402
from lrvla.vendor_sources import upstream_revision  # noqa: E402


def prepare(project: Path, *, resume_copy: bool = False) -> dict:
    project = project.resolve()
    source = project / "vendor/RoboTwin"
    destination = (project / "data/simulation/RoboTwin").resolve()
    backup = source.with_name("RoboTwin.pre-hpc")
    if source.is_symlink():
        if source.resolve() != destination or (destination / "assets").is_symlink():
            raise RuntimeError("Existing RoboTwin layout does not match the canonical scratch tree")
    else:
        if (destination.exists() and not resume_copy) or backup.exists():
            raise RuntimeError("Refusing to replace an existing source backup or scratch repository")
        destination.parent.mkdir(parents=True, exist_ok=True)
        def skip_top_assets(directory, names):
            return {"assets"} if Path(directory) == source else set()
        def copy_verified(original, copied):
            original, copied = Path(original), Path(copied)
            if copied.is_file():
                if original.stat().st_size == copied.stat().st_size:
                    def digest(path):
                        checksum = hashlib.sha256()
                        with path.open("rb") as stream:
                            for block in iter(lambda: stream.read(1024 * 1024), b""):
                                checksum.update(block)
                        return checksum.digest()
                    if digest(original) == digest(copied):
                        return str(copied)
                # Git pack files are read-only; this is only an unactivated
                # destination copy, so completing a partial file is safe.
                copied.chmod(copied.stat().st_mode | stat.S_IWUSR)
            return shutil.copy2(original, copied)
        shutil.copytree(source, destination, symlinks=True, dirs_exist_ok=resume_copy,
                        ignore=skip_top_assets, copy_function=copy_verified)
        assets = destination / "assets"
        source_assets = source / "assets"
        # Keep the active source and any external assets intact until the
        # complete destination has been verified, including during retries.
        if assets.is_symlink():
            assets.unlink()
        shutil.copytree(source_assets.resolve(strict=True), assets, symlinks=False,
                        dirs_exist_ok=resume_copy, copy_function=copy_verified)
        revision = upstream_revision(destination)
        if revision != ROBOTWIN_REVISION:
            raise RuntimeError(f"RoboTwin revision mismatch: {revision}")
        source.rename(backup)
        try:
            source.symlink_to(destination, target_is_directory=True)
        except BaseException:
            backup.rename(source)
            raise
    revision = upstream_revision(source)
    if revision != ROBOTWIN_REVISION or not (destination / "assets").is_dir():
        raise RuntimeError("Pinned RoboTwin source or its regular asset directory is invalid")
    return {"revision": revision, "source": str(source), "canonical_repository": str(destination),
            "assets": str(destination / "assets"), "assets_symlink": (destination / "assets").is_symlink(),
            "source_backup": str(backup), "git": (destination / ".git").exists(),
            "snapshot": (destination / ".lrvla_snapshot.json").exists()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--resume-copy", action="store_true", help="Resume this tool's interrupted, unactivated destination copy")
    args = parser.parse_args()
    report = prepare(args.project_root, resume_copy=args.resume_copy)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
