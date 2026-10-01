"""Packaged source remains verifiable and bootstrap never clones over it."""
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys

import pytest

from lrvla.vendor_sources import SNAPSHOT_NAME, snapshot_metadata, upstream_revision, verify_snapshot

ROOT = Path(__file__).resolve().parents[1]
REVISION = "a" * 40


def snapshot(tmp_path):
    repository = tmp_path / "upstream"
    (repository / "src").mkdir(parents=True)
    files = {"README.md": b"upstream documentation\n", "src/policy.py": b"value = 1\n"}
    for relative, content in files.items():
        (repository / relative).write_bytes(content)
    metadata = {"revision": REVISION, "files": {name: hashlib.sha256(content).hexdigest() for name, content in files.items()}}
    (repository / SNAPSHOT_NAME).write_text(json.dumps(metadata))
    return repository, metadata


def test_nongit_snapshot_supports_all_three_apis(tmp_path):
    repository, metadata = snapshot(tmp_path)
    assert not (repository / ".git").exists()
    assert verify_snapshot(repository, REVISION) == metadata
    assert snapshot_metadata(repository, REVISION) == metadata
    assert upstream_revision(repository) == REVISION


def test_all_checksums_are_checked_even_after_first_file_matches(tmp_path):
    repository, _ = snapshot(tmp_path)
    (repository / "src/policy.py").write_text("value = 2\n")
    with pytest.raises(ValueError, match="SHA256 mismatch: src/policy.py"):
        upstream_revision(repository)


def test_missing_file_and_wrong_revision_are_rejected(tmp_path):
    repository, _ = snapshot(tmp_path)
    with pytest.raises(ValueError, match="revision mismatch"):
        verify_snapshot(repository, "b" * 40)
    (repository / "src/policy.py").unlink()
    with pytest.raises(ValueError, match="missing or not a regular file"):
        verify_snapshot(repository, REVISION)


@pytest.mark.parametrize("name", ["../outside.py", "/etc/passwd", "C:/Windows/source.py", "src\\policy.py",
                                  ".git/config", "src//policy.py", "src/../policy.py", SNAPSHOT_NAME])
def test_unsafe_snapshot_paths_are_rejected(tmp_path, name):
    repository, _ = snapshot(tmp_path)
    (repository / SNAPSHOT_NAME).write_text(json.dumps({"revision": REVISION, "files": {name: "0" * 64}}))
    with pytest.raises(ValueError, match="Unsafe|cannot hash"):
        verify_snapshot(repository, REVISION)


def test_symlink_cannot_read_outside_source_tree(tmp_path):
    repository, metadata = snapshot(tmp_path)
    outside = tmp_path / "outside.py"
    outside.write_text("outside = 1\n")
    (repository / "src/policy.py").unlink()
    (repository / "src/policy.py").symlink_to(outside)
    metadata["files"]["src/policy.py"] = hashlib.sha256(outside.read_bytes()).hexdigest()
    (repository / SNAPSHOT_NAME).write_text(json.dumps(metadata))
    with pytest.raises(ValueError, match="escapes repository"):
        verify_snapshot(repository, REVISION)


def test_git_snapshot_tracks_sources_and_survives_copy_without_git(tmp_path):
    repository = tmp_path / "git-source"
    repository.mkdir()
    subprocess.run(["git", "init", "-q", str(repository)], check=True)
    (repository / "README.md").write_text("committed source\n")
    subprocess.run(["git", "-C", str(repository), "add", "README.md"], check=True)
    subprocess.run(["git", "-C", str(repository), "-c", "core.hooksPath=/dev/null", "-c", "commit.gpgsign=false",
                    "-c", "user.name=Vendor Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "source"], check=True)
    revision = upstream_revision(repository)
    (repository / "untracked.py").write_text("not in archive\n")
    metadata = snapshot_metadata(repository, revision)
    assert set(metadata["files"]) == {"README.md"}
    copied = tmp_path / "unpacked"
    copied.mkdir()
    shutil.copy2(repository / "README.md", copied / "README.md")
    (copied / SNAPSHOT_NAME).write_text(json.dumps(metadata))
    assert upstream_revision(copied) == revision
    with pytest.raises(ValueError, match="Git revision mismatch"):
        snapshot_metadata(repository, "0" * 40)


def bootstrap_source(repository, revision):
    script = (ROOT / "scripts/bootstrap.sh").read_text()
    function = script[script.index("prepare_upstream_source() {"):script.index("\nprepare_upstream_source vendor/")]
    # Only run the source-selection function, never the installer or network.
    program = ('test_python="$3"\npython() { "$test_python" "$@"; }\n'
               'git() { printf "Unexpected git call\\n" >&2; return 37; }\n' + function +
               '\nprepare_upstream_source "$1" unused-remote "$2"\n')
    return subprocess.run(["bash", "-c", program, "snapshot-bootstrap", str(repository), revision, sys.executable],
                          cwd=ROOT, text=True, capture_output=True, timeout=10)


def test_bootstrap_verifies_snapshot_without_git_clone(tmp_path):
    repository, _ = snapshot(tmp_path)
    result = bootstrap_source(repository, REVISION)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout)["verified_files"] == 2


def test_bootstrap_rejects_unverified_nonempty_directory(tmp_path):
    repository = tmp_path / "unverified"
    repository.mkdir()
    (repository / "README.md").write_text("unknown source\n")
    result = bootstrap_source(repository, REVISION)
    assert result.returncode == 1
    assert "no Git metadata or verified snapshot" in result.stderr
    assert "Unexpected git call" not in result.stderr
