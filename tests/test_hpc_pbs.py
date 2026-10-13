"""PBS jobs preserve assigned GPUs and literal arguments for one-node training."""
import importlib.util
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

import pytest


SOURCE = Path(__file__).parents[1] / "scripts/hpc/submit_pbs.py"
spec = importlib.util.spec_from_file_location("hpc_submit_pbs", SOURCE)
pbs = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pbs)


def arguments(root, **overrides):
    values = {"project": "260012", "queue": "normal", "workdir": str(root),
              "config": "configs/train_lora.yaml", "conda-root": str(root / "miniconda3")}
    values.update(overrides)
    result = []
    for key, value in values.items():
        result.extend(["--" + key, str(value)])
    return result


def test_default_only_generates_single_gpu_job(tmp_path, monkeypatch, capsys):
    def unexpected_submission(*args, **kwargs):
        raise AssertionError("Generating a job must not call a subprocess.")
    monkeypatch.setattr(pbs.subprocess, "run", unexpected_submission)
    assert pbs.main(arguments(tmp_path)) == 0
    report = json.loads(capsys.readouterr().out)
    assert not report["submitted"]
    script = Path(report["script"])
    assert script.stat().st_mode & 0o777 == 0o600
    content = script.read_text()
    assert "#PBS -l select=1:ngpus=1\n" in content
    assert "#PBS -l walltime=00:20:00\n" in content
    assert "#PBS -q normal\n" in content
    assert "export OMP_NUM_THREADS=4\n" in content
    assert "export MKL_NUM_THREADS=4\n" in content
    assert "export CUDA_VISIBLE_DEVICES" not in content
    assert "torchrun" not in content
    assert str(tmp_path / "artifacts/pbs/logs") in content


@pytest.mark.parametrize("key,value", [
    ("project", "260012\n#PBS -l select=1:ngpus=8"),
    ("project", "-Pother"),
    ("project", "260012;touch /tmp/pbs-pwn"),
    ("queue", "normal\n#PBS -q other"),
    ("queue", "aiq1"),
    ("walltime", "00:60:00"),
    ("walltime", "00:00:00"),
    ("walltime", "121:00:00"),
    ("walltime", "01:00:00\n#PBS -l select=2:ngpus=8"),
    ("workdir", "relative/repo"),
    ("workdir", "/tmp/repo\nexec malicious"),
    ("conda-root", "~/miniconda3"),
    ("conda-root", "/tmp/conda\x00invalid"),
    ("output-dir", "relative/logs"),
    ("config", "configs/a.yaml\nexec touch /tmp/pbs-pwn"),
    ("max-steps", "5; touch /tmp/pbs-pwn"),
    ("batch-size", "0"),
    ("gradient-accumulation", "-1"),
    ("global-batch-size", "0"),
    ("fallback-global-batch-size", "-1"),
    ("gpus", "2"),
    ("dependency", "123.pbs101; touch /tmp/pbs-pwn"),
    ("dependency", "afterok:123.pbs101\n#PBS -P other"),
    ("dependency", "afterany:123.pbs101"),
    ("dependency", "afterok:"),
    ("dependency", "123.pbs101:"),
])
def test_invalid_input_is_rejected_before_creating_files(tmp_path, key, value):
    with pytest.raises(SystemExit) as failure:
        pbs.main(arguments(tmp_path, **{key: value}))
    assert failure.value.code == 2
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("required", ["project", "queue", "workdir", "config", "conda-root"])
def test_scheduling_identity_and_paths_are_required(tmp_path, required):
    argv = arguments(tmp_path)
    offset = argv.index("--" + required)
    del argv[offset:offset + 2]
    with pytest.raises(SystemExit) as failure:
        pbs.main(argv)
    assert failure.value.code == 2


def test_submit_passes_exact_argv_without_shell(tmp_path, monkeypatch, capsys):
    calls = []
    def qsub(command, **kwargs):
        calls.append((command, kwargs))
        return subprocess.CompletedProcess(command, 0, "12345.pbs101\n", "")
    monkeypatch.setattr(pbs.subprocess, "run", qsub)
    directory = tmp_path / "logs with 'quotes' and $(touch forbidden)"
    assert pbs.main(arguments(tmp_path, **{"output-dir": directory}) + ["--submit"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["submitted"] and report["job_id"] == "12345.pbs101"
    assert calls == [(["qsub", "-mn", "-o", str(directory / "logs/lrvla.scheduler.log"), str(directory / "lrvla.pbs")],
                      {"capture_output": True, "text": True, "check": False})]


@pytest.mark.parametrize("dependency,expected", [
    ("123.pbs101", "afterok:123.pbs101"),
    ("afterok:123.pbs101:456.pbs102", "afterok:123.pbs101:456.pbs102"),
])
def test_submission_preserves_success_dependencies(tmp_path, monkeypatch, capsys, dependency, expected):
    calls = []
    def qsub(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0, "124.pbs101\n", "")
    monkeypatch.setattr(pbs.subprocess, "run", qsub)
    assert pbs.main(arguments(tmp_path, dependency=dependency) + ["--submit"]) == 0
    assert calls[0][:4] == ["qsub", "-mn", "-W", "depend=" + expected]
    assert json.loads(capsys.readouterr().out)["submitted"]


@pytest.mark.parametrize("queue,duration", [("normal", "24:00:00"), ("ai", "24:00:01"), ("ai", "02:00:00")])
def test_eight_gpu_request_rejects_incompatible_pool_before_writing(tmp_path, queue, duration):
    with pytest.raises(SystemExit) as failure:
        pbs.main(arguments(tmp_path, queue=queue, gpus=8, walltime=duration))
    assert failure.value.code == 2
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("overrides", [
    {"gpus": 1, "global-batch-size": 64},
    {"gpus": 4, "global-batch-size": 63},
    {"gpus": 4, "fallback-global-batch-size": 32},
    {"gpus": 4, "global-batch-size": 64, "fallback-global-batch-size": 31},
    {"gpus": 4, "global-batch-size": 64, "fallback-global-batch-size": 64},
    {"gpus": 4, "global-batch-size": 64, "fallback-global-batch-size": 128},
    {"gpus": 4, "global-batch-size": 64, "fallback-global-batch-size": 32, "batch-size": 16},
    {"gpus": 4, "global-batch-size": 64, "fallback-global-batch-size": 32, "gradient-accumulation": 1},
    {"gpus": 4, "global-batch-size": 64, "fallback-global-batch-size": 32, "resume": "runs/state.pt"},
])
def test_batch_fallback_rejects_inconsistent_training_options_before_writing(tmp_path, overrides):
    with pytest.raises(SystemExit) as failure:
        pbs.main(arguments(tmp_path, **overrides))
    assert failure.value.code == 2
    assert not (tmp_path / "artifacts").exists()


def test_submission_failure_propagates_and_does_not_claim_success(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(pbs.subprocess, "run", lambda command, **kwargs:
                        subprocess.CompletedProcess(command, 37, "", "Project does not have ai queue access"))
    assert pbs.main(arguments(tmp_path, queue="ai") + ["--submit"]) == 37
    output = capsys.readouterr()
    assert not output.out
    assert "Project does not have ai queue access" in output.err


def test_missing_qsub_reports_error(tmp_path, monkeypatch, capsys):
    def absent(*args, **kwargs):
        raise FileNotFoundError("qsub")
    monkeypatch.setattr(pbs.subprocess, "run", absent)
    assert pbs.main(arguments(tmp_path) + ["--submit"]) == 1
    assert "Cannot execute qsub" in capsys.readouterr().err


def fake_runtime(root, gpu_count=1):
    (root / "scripts").mkdir(parents=True)
    (root / "scripts/activate.sh").write_text(f"export CONDA_PREFIX={shlex.quote(str(root / 'conda'))}\n")
    (root / "conda/bin").mkdir(parents=True)
    (root / "conda/bin/python").symlink_to(sys.executable)
    (root / "torch.py").write_text(
        "from types import SimpleNamespace\n"
        "__version__ = 'stub-for-test'\n"
        "version = SimpleNamespace(cuda='stub')\n"
        f"cuda = SimpleNamespace(is_available=lambda: {bool(gpu_count)}, device_count=lambda: {gpu_count},\n"
        "get_device_name=lambda index: 'stub GPU',\n"
        "get_device_properties=lambda index: SimpleNamespace(total_memory=40 * 2**30),\n"
        "get_device_capability=lambda index: (8, 0))\n")
    train_code = ('import json, os, sys; print(json.dumps({"argv": sys.argv[1:], '
                  '"binding": os.environ.get("CUDA_VISIBLE_DEVICES")}))')
    (root / "train.sh").write_text(
        f'#!/bin/bash\n"${{CONDA_PREFIX}}/bin/python" -c {shlex.quote(train_code)} "$@"\n')
    torchrun = root / "conda/bin/torchrun"
    torchrun.write_text(
        f"#!{sys.executable}\n"
        "import json, os, sys\n"
        'print(json.dumps({"launcher": "torchrun", "argv": sys.argv[1:], '
        '"binding": os.environ.get("CUDA_VISIBLE_DEVICES")}))\n')
    torchrun.chmod(0o755)
    (root / "test-bin").mkdir()
    fake_smi = root / "test-bin/nvidia-smi"
    fake_smi.write_text("#!/bin/bash\nprintf 'Stub GPU driver for CPU-only test.\\n'\n")
    fake_smi.chmod(0o755)


def job_environment(root, job_id):
    return dict(os.environ, PBS_JOBID=job_id, CUDA_VISIBLE_DEVICES="GPU-assigned-by-PBS",
                PYTHONPATH=str(root), PATH=str(root / "test-bin") + os.pathsep + os.environ["PATH"])


def test_real_shell_keeps_paths_and_training_arguments_literal(tmp_path, capsys):
    root = tmp_path / "repo with 'quote' $(touch hacked) `touch also-hacked`"
    fake_runtime(root)
    config = "configs/config 'quoted' $(touch config-hacked) ; `touch config-backtick`.yaml"
    resume = "checkpoints/$(touch resume-hacked) 'quoted'.pt"
    run_output = root / "runs/output $(touch output-hacked)"
    assert pbs.main(arguments(root, **{"conda-root": root / "conda", "config": config, "resume": resume,
                                     "train-output-dir": run_output}) +
                    ["--allow-subset", "--max-steps", "2", "--batch-size", "8",
                     "--gradient-accumulation", "4", "--expert-last-n-layers", "-1"]) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    environment = job_environment(root, "42.pbs101")
    completed = subprocess.run(["/bin/bash", str(script)], env=environment, cwd=tmp_path,
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    log = (script.parent / "logs/42.pbs101.log").read_text()
    last_line = json.loads(log.splitlines()[-1])
    assert last_line == {"argv": ["--config", str(root / config),
                                 "--max-steps", "2", "--batch-size", "8", "--gradient-accumulation", "4",
                                 "--expert-last-n-layers", "-1", "--allow-subset", "--resume", str(root / resume),
                                 "--output-dir", str(run_output)], "binding": "GPU-assigned-by-PBS"}
    assert "PBS_JOBID=42.pbs101" in log
    assert "CUDA_VISIBLE_DEVICES=GPU-assigned-by-PBS" in log
    for name in ("hacked", "also-hacked", "config-hacked", "config-backtick", "resume-hacked", "output-hacked"):
        assert not (tmp_path / name).exists()
        assert not (root / name).exists()


def test_job_rejects_missing_pbs_identity_before_any_runtime(tmp_path, capsys):
    assert pbs.main(arguments(tmp_path)) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    environment = {key: value for key, value in os.environ.items() if key != "PBS_JOBID"}
    completed = subprocess.run(["/bin/bash", str(script)], env=environment, text=True, capture_output=True)
    assert completed.returncode != 0
    assert "not on a login node" in completed.stderr
    assert not list((script.parent / "logs").glob("*.log"))


@pytest.mark.parametrize("gpus,queue", [(4, "normal"), (8, "ai")])
def test_multigpu_job_uses_conda_torchrun_and_separate_output(tmp_path, capsys, gpus, queue):
    root = tmp_path / "repo with 'quoted' $(touch hacked)"
    fake_runtime(root, gpu_count=gpus)
    config = "configs/expert $(touch config-hacked).yaml"
    assert pbs.main(arguments(root, **{"gpus": gpus, "queue": queue, "config": config,
                                     "walltime": "24:00:00"})) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    content = script.read_text()
    assert f"#PBS -l select=1:ngpus={gpus}\n" in content
    assert "#PBS -m n\n" in content
    assert 'exec "${CONDA_PREFIX}/bin/torchrun" ' in content
    completed = subprocess.run(["/bin/bash", str(script)], env=job_environment(root, f"{gpus}.pbs101"),
                               cwd=tmp_path, text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    log = (script.parent / f"logs/{gpus}.pbs101.log").read_text()
    last_line = json.loads(log.splitlines()[-1])
    assert last_line == {
        "launcher": "torchrun",
        "argv": ["--standalone", "--nnodes=1", f"--nproc-per-node={gpus}",
                 str(root / "scripts/train.py"), "--config", str(root / config),
                 "--output-dir", str(root / f"runs/lrvla{gpus}_{gpus}gpu")],
        "binding": "GPU-assigned-by-PBS",
    }
    assert not (tmp_path / "hacked").exists()
    assert not (root / "config-hacked").exists()


def test_global_batch_without_fallback_keeps_direct_torchrun(tmp_path, capsys):
    root = tmp_path / "repo"
    fake_runtime(root, gpu_count=4)
    assert pbs.main(arguments(root, **{"gpus": 4, "global-batch-size": 64})) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    assert "run_with_batch_fallback.py" not in script.read_text()
    completed = subprocess.run(["/bin/bash", str(script)], env=job_environment(root, "46.pbs101"),
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    line = json.loads((script.parent / "logs/46.pbs101.log").read_text().splitlines()[-1])
    assert line["launcher"] == "torchrun"
    index = line["argv"].index("--global-batch-size")
    assert line["argv"][index + 1] == "64"


@pytest.mark.parametrize("global_batch,per_gpu", [(64, 16), (32, 8)])
def test_explicit_fixed_policy_runs_requested_batch_without_retry(tmp_path, capsys, global_batch, per_gpu):
    root = tmp_path / "repo"
    fake_runtime(root, gpu_count=4)
    assert pbs.main(arguments(root, **{
        "gpus": 4, "batch-policy": "fixed", "batch-size": per_gpu,
        "global-batch-size": global_batch, "gradient-accumulation": 1,
    })) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["batch_policy"] == "fixed"
    script = Path(report["script"])
    assert "run_with_batch_fallback.py" not in script.read_text()
    completed = subprocess.run(["/bin/bash", str(script)], env=job_environment(root, "48.pbs101"),
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    line = json.loads((script.parent / "logs/48.pbs101.log").read_text().splitlines()[-1])
    assert line["launcher"] == "torchrun"
    for flag, value in [("--batch-size", per_gpu), ("--global-batch-size", global_batch),
                        ("--gradient-accumulation", 1)]:
        assert line["argv"][line["argv"].index(flag) + 1] == str(value)


@pytest.mark.parametrize("overrides", [
    {"batch-policy": "fixed", "fallback-global-batch-size": 32, "global-batch-size": 64},
    {"batch-policy": "cuda-oom", "global-batch-size": 64},
    {"batch-policy": "unknown"},
])
def test_explicit_batch_policy_rejects_conflicting_options(tmp_path, overrides):
    with pytest.raises(SystemExit) as failure:
        pbs.main(arguments(tmp_path, gpus=4, **overrides))
    assert failure.value.code == 2
    assert not (tmp_path / "artifacts").exists()


@pytest.mark.parametrize("explicit", [False, True])
def test_cuda_oom_policy_preserves_implicit_launcher_compatibility(tmp_path, capsys, explicit):
    options = {"gpus": 4, "global-batch-size": 64, "fallback-global-batch-size": 32}
    if explicit:
        options["batch-policy"] = "cuda-oom"
    assert pbs.main(arguments(tmp_path, **options)) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["batch_policy"] == "cuda-oom"
    assert "run_with_batch_fallback.py" in Path(report["script"]).read_text()


def test_fallback_launcher_preserves_paths_and_forwarded_arguments(tmp_path, capsys):
    root = tmp_path / "repo with 'quote' $(touch hacked)"
    fake_runtime(root, gpu_count=4)
    launcher = root / "scripts/hpc/run_with_batch_fallback.py"
    launcher.parent.mkdir()
    launcher.write_text(
        "import json, os, sys\n"
        'print(json.dumps({"launcher": "batch-fallback", "argv": sys.argv[1:], '
        '"binding": os.environ.get("CUDA_VISIBLE_DEVICES")}))\n')
    config = "configs/expert 'quoted' $(touch config-hacked).yaml"
    output = root / "runs/out $(touch output-hacked)"
    assert pbs.main(arguments(root, **{"gpus": 4, "config": config, "train-output-dir": output,
                                     "global-batch-size": 64, "fallback-global-batch-size": 32,
                                     "max-steps": 1000, "expert-last-n-layers": -1}) + ["--allow-subset"]) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    assert 'exec "${CONDA_PREFIX}/bin/python" ' in script.read_text()
    completed = subprocess.run(["/bin/bash", str(script)], env=job_environment(root, "47.pbs101"),
                               cwd=tmp_path, text=True, capture_output=True, timeout=10)
    assert completed.returncode == 0, completed.stderr
    line = json.loads((script.parent / "logs/47.pbs101.log").read_text().splitlines()[-1])
    assert line == {
        "launcher": "batch-fallback",
        "argv": ["--gpus", "4", "--output-dir", str(output), "--config", str(root / config),
                 "--global-batch-size", "64", "--fallback-global-batch-size", "32", "--",
                 "--max-steps", "1000", "--expert-last-n-layers", "-1", "--allow-subset"],
        "binding": "GPU-assigned-by-PBS",
    }
    for name in ("hacked", "config-hacked", "output-hacked"):
        assert not (tmp_path / name).exists()
        assert not (root / name).exists()


@pytest.mark.parametrize("requested,visible", [(4, 1), (4, 8), (8, 4)])
def test_multigpu_job_rejects_gpu_count_mismatch(tmp_path, capsys, requested, visible):
    root = tmp_path / "repo"
    fake_runtime(root, gpu_count=visible)
    assert pbs.main(arguments(root, gpus=requested, queue="ai")) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    completed = subprocess.run(["/bin/bash", str(script)], env=job_environment(root, "45.pbs101"),
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode != 0
    log = (script.parent / "logs/45.pbs101.log").read_text()
    assert f"exactly {requested} PBS-assigned visible GPU" in log
    assert '"launcher"' not in log


@pytest.mark.parametrize("gpu_count", [0, 2])
def test_job_rejects_wrong_visible_gpu_count_before_training(tmp_path, capsys, gpu_count):
    root = tmp_path / "repo"
    fake_runtime(root, gpu_count=gpu_count)
    assert pbs.main(arguments(root)) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    completed = subprocess.run(["/bin/bash", str(script)], env=job_environment(root, "43.pbs101"),
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode != 0
    log = (script.parent / "logs/43.pbs101.log").read_text()
    assert f'"visible_gpu_count": {gpu_count}' in log
    assert "exactly one PBS-assigned visible GPU" in log
    assert '"argv"' not in log


def test_job_propagates_training_failure_to_pbs(tmp_path, capsys):
    root = tmp_path / "repo"
    fake_runtime(root)
    (root / "train.sh").write_text("#!/bin/bash\nprintf 'Training failed for CPU-only test.\\n'\nexit 43\n")
    assert pbs.main(arguments(root)) == 0
    script = Path(json.loads(capsys.readouterr().out)["script"])
    completed = subprocess.run(["/bin/bash", str(script)], env=job_environment(root, "44.pbs101"),
                               text=True, capture_output=True, timeout=10)
    assert completed.returncode == 43
    assert "Training failed" in (script.parent / "logs/44.pbs101.log").read_text()
