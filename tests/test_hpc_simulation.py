"""Simulation preparation preserves active assets and checks destination space."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import zipfile

import pytest

from lrvla.evaluation import ROBOTWIN_REVISION
from lrvla.vendor_sources import SNAPSHOT_NAME, upstream_revision

ROOT = Path(__file__).resolve().parents[1]


def load_script(name, relative):
    spec = importlib.util.spec_from_file_location(name, ROOT / relative)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


layout = load_script("hpc_simulation_layout", "scripts/hpc/prepare_simulation_layout.py")
assets = load_script("hpc_simulation_assets", "scripts/prepare_sim_assets.py")


def linked_source(tmp_path, revision=ROBOTWIN_REVISION):
    project = tmp_path / "project"
    source = project / "vendor/RoboTwin"
    source.mkdir(parents=True)
    external = tmp_path / "scratch/external-assets"
    external.mkdir(parents=True)
    (source / "assets").symlink_to(external, target_is_directory=True)
    files = {"README.md": b"pinned source\n", "assets/_download.py": b"pinned asset installer\n"}
    for name, content in files.items():
        (source / name).write_bytes(content)
    metadata = {"revision": revision, "files": {name: hashlib.sha256(content).hexdigest()
                                                for name, content in files.items()}}
    (source / SNAPSHOT_NAME).write_text(json.dumps(metadata))
    return project, source, external


def test_layout_keeps_old_assets_accessible_after_success_and_is_reusable(tmp_path):
    project, source, external = linked_source(tmp_path)
    report = layout.prepare(project)
    assert source.is_symlink()
    assert upstream_revision(source) == ROBOTWIN_REVISION
    assert not (source.resolve() / "assets").is_symlink()
    old_assets = source.with_name("RoboTwin.pre-hpc") / "assets/_download.py"
    assert old_assets.read_bytes() == (external / "_download.py").read_bytes()
    assert layout.prepare(project) == report


def test_failed_revision_check_keeps_active_assets_and_allows_copy_resume(tmp_path):
    project, source, external = linked_source(tmp_path, revision="a" * 40)
    with pytest.raises(RuntimeError, match="revision mismatch"):
        layout.prepare(project)
    assert source.is_dir() and not source.is_symlink()
    assert (source / "assets/_download.py").read_bytes() == (external / "_download.py").read_bytes()
    metadata = json.loads((source / SNAPSHOT_NAME).read_text())
    metadata["revision"] = ROBOTWIN_REVISION
    (source / SNAPSHOT_NAME).write_text(json.dumps(metadata))
    layout.prepare(project, resume_copy=True)
    assert upstream_revision(source) == ROBOTWIN_REVISION
    assert external.is_dir()


def test_failed_source_switch_restores_the_active_repository(tmp_path, monkeypatch):
    project, source, external = linked_source(tmp_path)
    def fail_switch(*args, **kwargs):
        raise OSError("source switch failed")
    monkeypatch.setattr(Path, "symlink_to", fail_switch)
    with pytest.raises(OSError, match="source switch failed"):
        layout.prepare(project)
    assert source.is_dir() and not source.is_symlink()
    assert (source / "assets/_download.py").read_bytes() == (external / "_download.py").read_bytes()
    assert not source.with_name("RoboTwin.pre-hpc").exists()


def asset_archive(tmp_path, monkeypatch):
    project = tmp_path / "home/project"
    (project / "artifacts").mkdir(parents=True)
    scratch_repo = tmp_path / "scratch/RoboTwin"
    destination = scratch_repo / "assets"
    destination.mkdir(parents=True)
    robotwin = project / "vendor/RoboTwin"
    robotwin.parent.mkdir()
    robotwin.symlink_to(scratch_repo, target_is_directory=True)
    archive_dir = project / "archives"
    archive_dir.mkdir()
    archive = archive_dir / "objects.zip"
    with zipfile.ZipFile(archive, "w") as bundle:
        bundle.writestr("completed.txt", b"already extracted")
        bundle.writestr("pending.txt", b"new")
    (destination / "completed.txt").write_bytes(b"already extracted")
    plan = {"sources": [{"files": [{"rfilename": archive.name,
                                    "lfs": {"sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}}]}]}
    (project / "artifacts/download_sim.json").write_text(json.dumps(plan))
    monkeypatch.setattr(assets, "ROOT", project)
    monkeypatch.setattr(sys, "argv", ["prepare_sim_assets.py", "--robotwin", str(robotwin),
                                     "--archives", str(archive_dir), "--files", "objects.zip"])
    return destination


def test_asset_space_check_uses_scratch_and_only_pending_members(tmp_path, monkeypatch):
    destination = asset_archive(tmp_path, monkeypatch)
    def disk_usage(path):
        assert Path(path) == destination
        return SimpleNamespace(free=3)
    monkeypatch.setattr(assets.shutil, "disk_usage", disk_usage)
    assets.main()
    assert (destination / "pending.txt").read_bytes() == b"new"
    monkeypatch.setattr(assets.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    assets.main()


def test_insufficient_scratch_space_fails_before_extracting_new_members(tmp_path, monkeypatch):
    destination = asset_archive(tmp_path, monkeypatch)
    monkeypatch.setattr(assets.shutil, "disk_usage", lambda path: SimpleNamespace(free=0))
    with pytest.raises(RuntimeError, match="Insufficient free disk space"):
        assets.main()
    assert not (destination / "pending.txt").exists()
    assert (destination / "completed.txt").read_bytes() == b"already extracted"
