"""Verify pinned upstream source from Git or a packaged SHA256 snapshot."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import subprocess
from .wsl_tools import linux_executable

SNAPSHOT_NAME = ".lrvla_snapshot.json"
REVISION = re.compile(r"[0-9a-f]{40}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def _revision(value: str) -> str:
    if not isinstance(value, str) or not REVISION.fullmatch(value):
        raise ValueError("Upstream revision must be a full lowercase 40-character Git SHA")
    return value


def _source_file(repository: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative or "\\" in relative or ":" in relative or "\0" in relative:
        raise ValueError(f"Unsafe snapshot path: {relative!r}")
    path = PurePosixPath(relative)
    if path.is_absolute() or path.as_posix() != relative or any(part in {".", "..", ".git"} for part in path.parts):
        raise ValueError(f"Unsafe snapshot path: {relative!r}")
    if relative == SNAPSHOT_NAME:
        raise ValueError("A snapshot cannot hash its own metadata")
    candidate = repository.joinpath(*path.parts)
    try:
        candidate.resolve().relative_to(repository)
    except ValueError as error:
        raise ValueError(f"Snapshot path escapes repository: {relative!r}") from error
    for current in (candidate, *candidate.parents):
        if current == repository:
            break
        if current.is_symlink():
            raise ValueError(f"Snapshot source must not be a symlink: {relative!r}")
    if not candidate.is_file():
        raise ValueError(f"Snapshot source is missing or not a regular file: {relative!r}")
    return candidate


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _snapshot(repository: Path) -> dict:
    path = repository / SNAPSHOT_NAME
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"Source directory has neither Git metadata nor a regular {SNAPSHOT_NAME}: {repository}")
    try:
        metadata = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"Invalid vendor snapshot metadata: {path}") from error
    if not isinstance(metadata, dict) or set(metadata) != {"revision", "files"}:
        raise ValueError("Vendor snapshot must contain exactly revision and files")
    _revision(metadata["revision"])
    if not isinstance(metadata["files"], dict) or not metadata["files"]:
        raise ValueError("Vendor snapshot must list at least one source file")
    return metadata


def _git_revision(repository: Path) -> str:
    try:
        git = str(linux_executable("git"))
        top = subprocess.check_output([git, "-C", str(repository), "rev-parse", "--show-toplevel"], text=True).strip()
        if Path(top).resolve() != repository:
            raise ValueError("Git metadata does not belong to the requested upstream directory")
        result = subprocess.check_output([git, "-C", str(repository), "rev-parse", "HEAD"], text=True).strip()
    except subprocess.CalledProcessError as error:
        raise ValueError(f"Cannot read upstream Git revision: {repository}") from error
    return _revision(result)


def verify_snapshot(repository: Path, expected_revision: str) -> dict:
    """Validate the pinned revision and every packaged source checksum."""
    repository = Path(repository).resolve()
    expected_revision = _revision(expected_revision)
    metadata = _snapshot(repository)
    if metadata["revision"] != expected_revision:
        raise ValueError(f"Vendor snapshot revision mismatch: {metadata['revision']} != {expected_revision}")
    for relative, checksum in metadata["files"].items():
        if not isinstance(checksum, str) or not SHA256.fullmatch(checksum):
            raise ValueError(f"Invalid snapshot SHA256 for {relative!r}")
        source = _source_file(repository, relative)
        if _sha256(source) != checksum:
            raise ValueError(f"Vendor snapshot SHA256 mismatch: {relative}")
    return metadata


def snapshot_metadata(repository: Path, expected_revision: str) -> dict:
    """Hash Git-tracked files, or verify and reuse an existing packaged snapshot."""
    repository = Path(repository).resolve()
    expected_revision = _revision(expected_revision)
    if not (repository / ".git").exists():
        return verify_snapshot(repository, expected_revision)
    actual = _git_revision(repository)
    if actual != expected_revision:
        raise ValueError(f"Upstream Git revision mismatch: {actual} != {expected_revision}")
    tracked = subprocess.check_output([str(linux_executable("git")), "-C", str(repository), "ls-files", "--stage", "-z"])
    files = {}
    for entry in filter(None, tracked.split(b"\0")):
        details, name = entry.split(b"\t", 1)
        mode, _, stage = details.split(b" ")
        if stage != b"0" or mode not in {b"100644", b"100755"}:
            raise ValueError("Vendor snapshot requires regular tracked files without merge conflicts or submodules")
        relative = name.decode("utf-8")
        if relative == SNAPSHOT_NAME:
            continue
        files[relative] = _sha256(_source_file(repository, relative))
    if not files:
        raise ValueError("Upstream Git repository contains no tracked source files")
    return {"revision": actual, "files": dict(sorted(files.items()))}


def upstream_revision(repository: Path) -> str:
    """Return Git HEAD, or the revision of a fully verified packaged snapshot."""
    repository = Path(repository).resolve()
    if (repository / ".git").exists():
        return _git_revision(repository)
    metadata = _snapshot(repository)
    return verify_snapshot(repository, metadata["revision"])["revision"]
