"""Remote policy servers with local simulators: server manifest, client scheduling and quick-evaluation output."""
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import threading
import time

import pytest

from lrvla import quick_evaluation as quick
from lrvla import remote_evaluation as remote
from lrvla.data import TASKS

ROOT = Path(__file__).resolve().parents[1]
LIMITS = {task: 400 + 100 * (index % 7) for index, task in enumerate(TASKS)}
FAKE_SERVER = '''import argparse, socket, time
parser = argparse.ArgumentParser()
for name in ("--model_path", "--use_length", "--use_bf16", "--use_fp32", "--use_compile", "--port", "--num_steps"):
    parser.add_argument(name)
args = parser.parse_args()
assert args.use_bf16 == "True" and args.use_fp32 == "False" and args.num_steps == "10"
listener = socket.socket(); listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
listener.bind(("127.0.0.1", int(args.port))); listener.listen()
while True:
    time.sleep(1)
'''
FAKE_CLIENT = '''import argparse, json, os, sys
from pathlib import Path
parser = argparse.ArgumentParser()
parser.add_argument("--config"); parser.add_argument("--overrides", nargs=argparse.REMAINDER)
args = parser.parse_args()
values = dict(zip(args.overrides[0::2], args.overrides[1::2]))
task, setting, port = values["--task_name"], values["--task_config"], values["--port"]
marker = Path(os.environ["FAKE_MARKERS"]) / f"{setting}-{task}"
if task == "adjust_bottle" and setting == "demo_randomized" and not marker.exists():
    marker.parent.mkdir(parents=True, exist_ok=True); marker.write_text("failed once"); sys.exit(1)
successes = (len(task) + (0 if setting == "demo_clean" else 3)) % 11
print("LRVLA_EPISODE_METRICS=" + json.dumps({"expert_seconds": 1.0, "total_seconds": 4.0, "port": int(port)}))
for scene in range(1, 11):
    print(f"Success rate: {min(scene, successes)}/{scene} => 0%, current seed: {scene}")
'''


def free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def listener(port):
    server = socket.socket()
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind(("127.0.0.1", port))
    server.listen()
    threading.Thread(target=lambda: [server.accept()[0].close() for _ in iter(int, 1)], daemon=True).start()
    return server


def manifest(tmp_path, ports, **override):
    record = {"schema": remote.MANIFEST_SCHEMA, "host": "gpu-node-1", "started_at": "now", "checkpoint": "/cluster/run/hf_ckpt",
              "checkpoint_identity": {"sha256": "weights"}, "endpoints": [{"port": port, "gpu": str(i)} for i, port in enumerate(ports)],
              **remote.PROFILE, **override}
    path = tmp_path / "servers.json"
    path.write_text(json.dumps(record))
    return path


def test_servers_publish_a_manifest_only_when_all_listen_and_stop_together(tmp_path):
    run = tmp_path / "run"
    checkpoint = run / "checkpoints/global_step_1/hf_ckpt"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model-00001.safetensors").write_bytes(b"weights")
    (tmp_path / "fake_server.py").write_text(FAKE_SERVER)
    ports = [free_port(), free_port()]
    code = (f"import sys; sys.path.insert(0, {str(ROOT / 'src')!r})\n"
            "from lrvla.remote_evaluation import serve\n"
            f"serve({str(checkpoint)!r}, {str(tmp_path / 'servers.json')!r}, {ports!r}, ['0', '1'], "
            f"script={str(tmp_path / 'fake_server.py')!r}, timeout=60)\n")
    process = subprocess.Popen([sys.executable, "-c", code], start_new_session=True)
    try:
        for _ in range(100):
            if (tmp_path / "servers.json").is_file():
                break
            time.sleep(0.2)
        record = json.loads((tmp_path / "servers.json").read_text())
        assert all(remote.port_open("127.0.0.1", port) for port in ports)
        assert [endpoint["port"] for endpoint in record["endpoints"]] == ports
        assert record["checkpoint_identity"]["shards"][0]["file"] == "model-00001.safetensors"
        assert {key: record[key] for key in remote.PROFILE} == remote.PROFILE
        os.killpg(process.pid, signal.SIGTERM)
        assert process.wait(timeout=60) == 0
        assert not any(remote.port_open("127.0.0.1", port) for port in ports)
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)


def test_simulators_share_one_queue_retry_failures_and_write_the_quick_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(quick, "step_limits", lambda robotwin: dict(LIMITS))
    monkeypatch.setenv("FAKE_MARKERS", str(tmp_path / "markers"))
    client = tmp_path / "fake_client.py"
    client.write_text(FAKE_CLIENT)
    ports = [free_port(), free_port(), free_port()]
    servers = [listener(port) for port in ports]
    try:
        code = remote.evaluate(manifest(tmp_path, ports), [("127.0.0.1", port) for port in ports], tmp_path / "out",
                               client=client, python=sys.executable, preflight=False, progress_interval=1)
    finally:
        for server in servers:
            server.close()
    assert code == 0
    out = tmp_path / "out"
    summary = json.loads((out / "summary.json").read_text())
    assert summary["complete"] and summary["scenes"] == 1000 and summary["policy_rng"] == ["legacy"]
    assert summary["weights_sha256"] == "weights" and summary["profile"]["precision"] == "bf16"
    attempts = json.loads((out / "status.json").read_text())
    retried = [row for row in attempts if row["task"] == "adjust_bottle" and row["setting"] == "demo_randomized"]
    assert [row["exit_code"] for row in retried] == [1, 0]
    assert {row["slot"] for row in attempts} == {0, 1, 2}
    plan = json.loads((out / "shards/s0/evaluation_plan.json").read_text())
    assert plan["remote_inference"]["server_host"] == "gpu-node-1" and plan["sim_workers"] == 3
    # Everything is complete, so a rerun starts no simulator.
    assert remote.run_clients(out, json.loads((out / "quick_plan.json").read_text()), [("127.0.0.1", 1)], client=client,
                              python=sys.executable, env=dict(os.environ), cwd=tmp_path, limits=LIMITS)
    assert len(json.loads((out / "status.json").read_text())) == len(attempts)


def test_unreachable_endpoint_stops_its_simulator_and_keeps_the_work(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_MARKERS", str(tmp_path / "markers"))
    client = tmp_path / "fake_client.py"
    client.write_text(FAKE_CLIENT)
    record = remote.load_manifest(manifest(tmp_path, [9]))
    out, plan = remote.prepare_output(tmp_path / "out", record, [("127.0.0.1", free_port())])
    assert remote.run_clients(out, plan, [("127.0.0.1", free_port())], client=client, python=sys.executable,
                              env=dict(os.environ), cwd=tmp_path, limits=LIMITS, endpoint_wait=0, progress_interval=1)
    assert not (out / "status.json").exists()
    assert not any((out / "shards/s0").rglob("*.log"))


def test_protocol_and_checkpoint_guards(tmp_path):
    with pytest.raises(ValueError):
        remote.load_manifest(manifest(tmp_path, [1], precision="fp32"))
    with pytest.raises(ValueError):
        remote.load_manifest(manifest(tmp_path, [1], num_steps=5))
    record = remote.load_manifest(manifest(tmp_path, [1]))
    remote.prepare_output(tmp_path / "out", record, [("127.0.0.1", 1)])
    remote.prepare_output(tmp_path / "out", record, [("127.0.0.1", 1), ("127.0.0.1", 2)])   # resume with more simulators
    with pytest.raises(ValueError):
        remote.prepare_output(tmp_path / "out", {**record, "checkpoint_identity": {"sha256": "other"}}, [("127.0.0.1", 1)])
    assert remote.parse_endpoints("127.0.0.1:9601,localhost:9602") == [("127.0.0.1", 9601), ("localhost", 9602)]
    for bad in ("9601", "host:", "a:1,a:1"):
        with pytest.raises(Exception):
            remote.parse_endpoints(bad)
    command = remote.server_command("/run/hf_ckpt", 9601, python="py", script="serve_native.py")
    assert command[command.index("--use_bf16") + 1] == "True" and command[command.index("--num_steps") + 1] == "10"
