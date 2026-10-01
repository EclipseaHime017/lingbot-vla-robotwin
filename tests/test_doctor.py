"""The doctor must survive failed/blocked native probes and report readiness honestly."""
import importlib.util
import hashlib
import json
from pathlib import Path
import sys


spec = importlib.util.spec_from_file_location('project_doctor', Path(__file__).parents[1] / 'scripts/doctor.py')
doctor = importlib.util.module_from_spec(spec)
spec.loader.exec_module(doctor)


def test_native_process_timeout_is_bounded():
    result = doctor.isolated_probe(Path(sys.executable), 'import time; time.sleep(5)', timeout=0.05)
    assert result['status'] == 'timeout'
    assert not result['ok']
    assert result['seconds'] < 3


def test_native_crash_is_recorded():
    result = doctor.isolated_probe(Path(sys.executable), 'import os; os.abort()', timeout=3)
    assert result['status'] == 'failed'
    assert result['returncode'] < 0


def test_cuda_and_physics_do_not_imply_rendering_ready():
    report = {'host': {'wsl': True}, 'sim': {'cuda': {'ok': True}, 'physics': {'ok': True},
              'render': {'ok': False}}, 'assets': {'aloha_agilex': True}}
    summary = doctor.summarize(report)
    assert summary['simulation_cuda'] and summary['cpu_physics']
    assert not summary['camera_rendering'] and not summary['closed_loop_ready']
    assert any('native Linux' in message for message in summary['messages'])


def test_manifest_classification_does_not_invent_nvidia_driver(tmp_path):
    (tmp_path / 'lvp_icd.json').write_text(json.dumps({'ICD': {'library_path': 'libvulkan_lvp.so'}}))
    result = doctor.vulkan_icds([tmp_path])
    assert len(result) == 1
    assert not result[0]['nvidia']


def test_probe_emits_structured_result():
    result = doctor.isolated_probe(Path(sys.executable),
        'print(\'DOCTOR_JSON={"answer": 42}\')', timeout=3)
    assert result['ok']
    assert result['details']['answer'] == 42


def test_early_native_exit_is_not_a_successful_probe():
    result = doctor.isolated_probe(Path(sys.executable), 'raise SystemExit(0)', timeout=3)
    assert not result['ok']
    assert result['status'] == 'missing_probe_output'


def test_native_probe_preserves_shell_tool_and_library_configuration():
    configuration = {'CC': '/mnt/c/Tools/cl.exe', 'CUDA_HOME': 'C:\\CUDA',
                     'LD_LIBRARY_PATH': '/mnt/c/Tools/lib:/usr/lib',
                     'PATH': '/usr/bin:/mnt/c/Windows/System32'}
    result = doctor.isolated_probe(Path(sys.executable),
        'import os, json; print("DOCTOR_JSON=" + json.dumps({"cc": os.getenv("CC"), '
        '"cuda": os.getenv("CUDA_HOME"), "libs": os.getenv("LD_LIBRARY_PATH"), '
        '"path": os.getenv("PATH")}))', timeout=3, env=configuration)
    assert result['ok']
    assert result['details'] == {'cc': configuration['CC'], 'cuda': configuration['CUDA_HOME'],
                                 'libs': configuration['LD_LIBRARY_PATH'], 'path': configuration['PATH']}


def test_native_probe_rejects_non_linux_interpreter_before_launch(tmp_path):
    python = tmp_path / 'python'
    python.write_bytes(b'MZ' + bytes(128))
    python.chmod(0o755)
    result = doctor.isolated_probe(python, 'raise SystemExit(0)', timeout=3)
    assert not result['ok']
    assert result['status'] == 'invalid_executable'
    assert result['python'] == str(python)


def test_upstream_revision_is_reported_and_integrity_errors_are_not_ready(monkeypatch, tmp_path):
    monkeypatch.setattr(doctor, 'upstream_revision', lambda repo: 'expected')
    assert doctor.revision_status(tmp_path, 'expected')['ok']
    assert doctor.revision_status(tmp_path, 'other')['status'] == 'revision_mismatch'
    def invalid_source(repo):
        raise ValueError('snapshot file digest changed')
    monkeypatch.setattr(doctor, 'upstream_revision', invalid_source)
    status = doctor.revision_status(tmp_path, 'expected')
    assert not status['ok'] and status['status'] == 'unverified_source'
    assert 'digest changed' in status['error']


def test_simulation_readiness_requires_verified_source():
    report = {'sim': {key: {'ok': True} for key in
                     ('imports', 'cuda', 'physics', 'render', 'curobo', 'robotwin_import')},
              'assets': {key: True for key in ('aloha_agilex', 'objects', 'textures')},
              'upstream': {'robotwin': {'ok': False}}}
    assert not doctor.summarize(report)['closed_loop_ready']
    report['upstream']['robotwin']['ok'] = True
    assert doctor.summarize(report)['closed_loop_ready']


def test_doctor_accepts_verified_zip_source_and_detects_later_changes(tmp_path):
    source = tmp_path / 'source.py'
    source.write_text('VALUE = 1\n')
    (tmp_path / '.lrvla_snapshot.json').write_text(json.dumps({
        'revision': doctor.EXPECTED_ROBOTWIN_REVISION,
        'files': {'source.py': hashlib.sha256(source.read_bytes()).hexdigest()}}))
    assert doctor.revision_status(tmp_path, doctor.EXPECTED_ROBOTWIN_REVISION)['ok']
    source.write_text('VALUE = 2\n')
    result = doctor.revision_status(tmp_path, doctor.EXPECTED_ROBOTWIN_REVISION)
    assert not result['ok'] and result['status'] == 'unverified_source'
