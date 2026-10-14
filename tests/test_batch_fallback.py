"""Physical batch fallback reacts to CUDA OOM and cleans up only its ranks."""
import argparse
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest


SOURCE = Path(__file__).parents[1] / "scripts/run_with_batch_fallback.py"
spec = importlib.util.spec_from_file_location("batch_fallback", SOURCE)
launcher = importlib.util.module_from_spec(spec)
spec.loader.exec_module(launcher)


def arguments(tmp_path):
    config = tmp_path / "config 'literal' $(touch forbidden).yaml"
    config.write_text("mode: full\n")
    return ["--gpus", "4", "--output-dir", str(tmp_path / "runs"), "--config", str(config),
            "--global-batch-size", "64", "--fallback-global-batch-size", "32"]


@pytest.fixture
def child_launcher(tmp_path, monkeypatch):
    child = tmp_path / "fake_train.py"
    child.write_text('''import json, os
from pathlib import Path
import signal
import subprocess
import sys
import time

batch, output, mode = int(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
if mode == 'stubborn' and batch == 16:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
output.mkdir(parents=True, exist_ok=True)
(output / 'argv.json').write_text(json.dumps(sys.argv[4:]))
(output / 'checkpoint.txt').write_text('independent attempt ' + str(batch))
if mode == 'ordinary-error':
    print('CPU out of memory; unrelated failure', flush=True)
    raise SystemExit(37)
if mode == 'group' and batch == 16:
    helper = subprocess.Popen([sys.executable, '-c', "import signal,time; signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(60)"])
    (output / 'rank.pid').write_text(str(helper.pid))
    time.sleep(.1)
if mode == 'both-oom' or (mode in ('oom-then-success', 'cuda-text', 'group', 'stubborn') and batch == 16):
    if mode == 'cuda-text':
        print('torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1 GiB', flush=True)
    else:
        print(json.dumps({'event': 'LRVLA_CUDA_OOM', 'rank': 0}), flush=True)
    time.sleep(60)
print('training completed with batch ' + str(batch), flush=True)
''')
    state = {"mode": "success"}
    def build_command(args, batch_size, run_dir):
        return [sys.executable, str(child), str(batch_size), str(run_dir), state["mode"],
                *args.training_args]
    monkeypatch.setattr(launcher, "build_command", build_command)
    return state


def report(tmp_path):
    return json.loads((tmp_path / "runs/attempts.json").read_text())


def test_success_keeps_first_batch_and_literal_arguments(tmp_path, child_launcher):
    extra = ["--max-steps", "1000", "--expert-last-n-layers", "-1", "--allow-subset"]
    assert launcher.main(arguments(tmp_path) + ["--", *extra]) == 0
    actual = report(tmp_path)
    assert actual["status"] == "succeeded"
    assert actual["first_attempt_global_batch_size"] == 64
    assert actual["actual_global_batch_size"] == 64
    assert actual["fallback_used"] is False
    assert len(actual["attempts"]) == 1
    attempt = actual["attempts"][0]
    assert attempt["batch_size_per_gpu"] == 16 and attempt["gradient_accumulation"] == 1
    assert not attempt["cuda_oom"] and attempt["exit_code"] == 0
    assert json.loads((tmp_path / "runs/batch64/argv.json").read_text()) == extra
    assert "training completed" in (tmp_path / "runs/batch64.launcher.log").read_text()
    assert not (tmp_path / "runs/batch32").exists()
    assert not (tmp_path / "forbidden").exists()


@pytest.mark.parametrize("mode", ["oom-then-success", "cuda-text"])
def test_clear_cuda_oom_retries_once_with_independent_output(tmp_path, child_launcher, mode):
    child_launcher["mode"] = mode
    assert launcher.main(arguments(tmp_path)) == 0
    actual = report(tmp_path)
    assert actual["status"] == "succeeded" and actual["fallback_used"]
    assert actual["actual_global_batch_size"] == 32
    assert [attempt["batch_size_per_gpu"] for attempt in actual["attempts"]] == [16, 8]
    assert [attempt["gradient_accumulation"] for attempt in actual["attempts"]] == [1, 1]
    assert [attempt["cuda_oom"] for attempt in actual["attempts"]] == [True, False]
    assert (tmp_path / "runs/batch64/checkpoint.txt").read_text() == "independent attempt 16"
    assert (tmp_path / "runs/batch32/checkpoint.txt").read_text() == "independent attempt 8"
    assert "out of memory" in (tmp_path / "runs/batch64.launcher.log").read_text() or (
        "LRVLA_CUDA_OOM" in (tmp_path / "runs/batch64.launcher.log").read_text())


def test_non_cuda_error_never_retries(tmp_path, child_launcher):
    child_launcher["mode"] = "ordinary-error"
    assert launcher.main(arguments(tmp_path)) == 37
    actual = report(tmp_path)
    assert actual["status"] == "failed" and not actual["fallback_used"]
    assert len(actual["attempts"]) == 1
    assert actual["attempts"][0]["exit_code"] == 37
    assert not actual["attempts"][0]["cuda_oom"]
    assert not (tmp_path / "runs/batch32").exists()


def test_second_oom_fails_without_third_attempt(tmp_path, child_launcher):
    child_launcher["mode"] = "both-oom"
    assert launcher.main(arguments(tmp_path)) != 0
    actual = report(tmp_path)
    assert actual["status"] == "failed" and actual["fallback_used"]
    assert len(actual["attempts"]) == 2
    assert all(attempt["cuda_oom"] for attempt in actual["attempts"])


def process_active(pid):
    try:
        return Path(f"/proc/{pid}/stat").read_text().split(")", 1)[1].split()[0] != "Z"
    except FileNotFoundError:
        return False


def test_oom_kills_rank_group_and_keeps_unrelated_process(tmp_path, child_launcher):
    child_launcher["mode"] = "group"
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"],
                                 start_new_session=True)
    try:
        assert launcher.main(arguments(tmp_path)) == 0
        rank_pid = int((tmp_path / "runs/batch64/rank.pid").read_text())
        deadline = time.monotonic() + 2
        while process_active(rank_pid) and time.monotonic() < deadline:
            time.sleep(.02)
        assert not process_active(rank_pid)
        assert unrelated.poll() is None
        first = report(tmp_path)["attempts"][0]
        assert first["process_group"] != os.getpgrp()
    finally:
        unrelated.terminate()
        unrelated.wait(timeout=2)


def test_training_command_uses_current_python_and_physical_batch(tmp_path):
    args = argparse.Namespace(gpus=4, config=tmp_path / "config.yaml", training_args=["--max-steps", "1000"])
    run_dir = tmp_path / "batch64"
    assert launcher.build_command(args, 16, run_dir) == [
        sys.executable, "-m", "torch.distributed.run", "--standalone", "--nnodes=1", "--nproc-per-node=4",
        str(launcher.PROJECT_ROOT / "scripts/train.py"), "--config", str(args.config),
        "--batch-size", "16", "--global-batch-size", "64", "--gradient-accumulation", "1", "--output-dir", str(run_dir),
        "--max-steps", "1000"]


def test_oom_kills_group_that_ignores_sigterm_after_grace(tmp_path, child_launcher, monkeypatch):
    child_launcher["mode"] = "stubborn"
    monkeypatch.setattr(launcher, "TERMINATE_GRACE_SECONDS", .1)
    started = time.monotonic()
    assert launcher.main(arguments(tmp_path)) == 0
    assert time.monotonic() - started < 2
    first = report(tmp_path)["attempts"][0]
    assert first["cuda_oom"] and first["exit_code"] == -9
    assert not process_active(first["process_group"])


@pytest.mark.parametrize("option", ["--batch-size", "--global-batch-size", "--gradient-accumulation", "--output-dir", "--config", "--resume", "--res", "--batch", "--output"])
@pytest.mark.parametrize("equals", [False, True])
def test_duplicate_controlled_training_arguments_rejected(tmp_path, option, equals):
    duplicate = [option + "=8"] if equals else [option, "8"]
    with pytest.raises(SystemExit) as failure:
        launcher.main(arguments(tmp_path) + ["--", *duplicate])
    assert failure.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_yaml_resume_rejected_before_starting_or_creating_output(tmp_path):
    argv = arguments(tmp_path)
    config = Path(argv[argv.index("--config") + 1])
    config.write_text("mode: expert\nresume: /old/checkpoint\n")
    with pytest.raises(SystemExit) as failure:
        launcher.main(argv)
    assert failure.value.code == 2
    assert not (tmp_path / "runs").exists()


@pytest.mark.parametrize("option,value", [("--gpus", "0"), ("--global-batch-size", "63"),
                                         ("--fallback-global-batch-size", "64"),
                                         ("--fallback-global-batch-size", "0"),
                                         ("--output-dir", "relative"), ("--config", "relative.yaml")])
def test_invalid_launch_budget_rejected_before_output(tmp_path, option, value):
    argv = arguments(tmp_path)
    argv[argv.index(option) + 1] = value
    with pytest.raises(SystemExit) as failure:
        launcher.main(argv)
    assert failure.value.code == 2
    assert not (tmp_path / "runs").exists()


def test_existing_attempts_are_preserved(tmp_path, child_launcher):
    argv = arguments(tmp_path)
    assert launcher.main(argv) == 0
    before = (tmp_path / "runs/attempts.json").read_bytes()
    with pytest.raises(SystemExit):
        launcher.main(argv)
    assert (tmp_path / "runs/attempts.json").read_bytes() == before


def test_real_cli_uses_same_python_and_torchrun_arguments_without_gpu(tmp_path):
    fake_modules = tmp_path / "fake-modules"
    package = fake_modules / "torch/distributed"
    package.mkdir(parents=True)
    (fake_modules / "torch/__init__.py").write_text("")
    (package / "__init__.py").write_text("")
    (package / "run.py").write_text('''import argparse, json
from pathlib import Path
import sys, time
parser = argparse.ArgumentParser()
parser.add_argument('--standalone', action='store_true')
parser.add_argument('--nnodes', type=int)
parser.add_argument('--nproc-per-node', type=int)
parser.add_argument('train_script')
parser.add_argument('--config')
parser.add_argument('--batch-size', type=int)
parser.add_argument('--global-batch-size', type=int)
parser.add_argument('--gradient-accumulation', type=int)
parser.add_argument('--output-dir', type=Path)
parser.add_argument('--max-steps', type=int)
args = parser.parse_args()
assert args.standalone and args.nnodes == 1 and args.nproc_per_node == 4
assert args.gradient_accumulation == 1
assert args.global_batch_size == args.batch_size * 4
assert args.max_steps == 1000
(args.output_dir / 'actual-cli.json').write_text(json.dumps({'python':sys.executable, 'batch':args.batch_size, 'global':args.global_batch_size}))
if args.global_batch_size == 64:
    print('LRVLA_CUDA_OOM rank=0', flush=True)
    time.sleep(60)
print('Completed actual fallback CLI', flush=True)
''')
    environment = dict(os.environ, PYTHONPATH=str(fake_modules))
    result = subprocess.run([sys.executable, str(SOURCE), *arguments(tmp_path), "--", "--max-steps", "1000"],
                            env=environment, capture_output=True, text=True, timeout=5)
    assert result.returncode == 0, result.stderr
    actual = report(tmp_path)
    assert actual["fallback_used"] and actual["actual_global_batch_size"] == 32
    for global_batch, per_gpu in ((64, 16), (32, 8)):
        recorded = json.loads((tmp_path / f"runs/batch{global_batch}/actual-cli.json").read_text())
        assert recorded == {"python": sys.executable, "batch": per_gpu, "global": global_batch}
