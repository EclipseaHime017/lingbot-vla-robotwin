"""Exercise OpenSSH/askpass boundaries with synthetic credentials only."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import signal
import sys
import time

import pytest


SOURCE = Path(__file__).parents[1] / "scripts/hpc/connect_ssh.py"
spec = importlib.util.spec_from_file_location("hpc_connect_ssh", SOURCE)
connector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(connector)
SECRET = "  synthetic $(touch forbidden) `touch forbidden` ; secret  "


@pytest.fixture
def credentials(tmp_path):
    path = tmp_path / "credentials with spaces.txt"
    path.write_text(f"Account explanation\nssh m260012@aspire2antu.nscc.sg\npassword {SECRET}\n")
    path.chmod(0o600)
    return path


@pytest.fixture
def fake_ssh(tmp_path, monkeypatch, credentials):
    control = tmp_path / "control"
    control.mkdir(mode=0o700)
    log = tmp_path / "ssh.jsonl"
    fake = tmp_path / "ssh"
    fake.write_text(f"#!{sys.executable}\n" + '''import hashlib, json, os
from pathlib import Path
import stat
import subprocess
import sys

args = sys.argv[1:]
entry = {'argv': args}
if '-G' in args:
    print('user m260012\\nhostname aspire2antu.nscc.sg\\ncontrolpath ' + os.environ['TEST_CONTROL'])
    code = 0
elif '-O' in args:
    code = 0 if os.environ.get('TEST_REUSE') == '1' else 255
else:
    secret = Path(os.environ['TEST_CREDENTIALS']).read_text().split('password ', 1)[1].rstrip('\\n')
    entry['secret_in_argv_or_env'] = secret in json.dumps(args) or any(secret in v for v in os.environ.values())
    helper = Path(os.environ['SSH_ASKPASS'])
    entry['helper_mode'] = stat.S_IMODE(helper.stat().st_mode)
    entry['directory_mode'] = stat.S_IMODE(helper.parent.stat().st_mode)
    entry['helper_path'] = str(helper)
    entry['isolated_session'] = os.getsid(0) == os.getpid()
    answer = subprocess.run([str(helper), os.environ.get('TEST_PROMPT', "m260012@aspire2antu.nscc.sg's password:")],
                            capture_output=True, check=False)
    entry['askpass_exit'] = answer.returncode
    entry['password_sha256'] = hashlib.sha256(answer.stdout).hexdigest()
    code = int(os.environ.get('TEST_AUTH_EXIT', '0')) if answer.returncode == 0 else 255
    if '-MNf' in args and os.environ.get('TEST_DAEMON') == '1':
        daemon = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(5)'])
        entry['daemon_pid'] = daemon.pid
with Path(os.environ['TEST_LOG']).open('a') as stream:
    stream.write(json.dumps(entry) + '\\n')
raise SystemExit(code)
''')
    fake.chmod(0o700)
    monkeypatch.setattr(connector, "SSH", str(fake))
    monkeypatch.setenv("TEST_CREDENTIALS", str(credentials))
    monkeypatch.setenv("TEST_CONTROL", str(control / "socket"))
    monkeypatch.setenv("TEST_LOG", str(log))
    return log


def entries(log):
    return [json.loads(line) for line in log.read_text().splitlines()]


def options(argv):
    return [argv[i + 1] for i, value in enumerate(argv[:-1]) if value == "-o"]


def test_existing_master_is_reused_without_askpass(credentials, fake_ssh, monkeypatch, capsys):
    monkeypatch.setenv("TEST_REUSE", "1")
    assert connector.main(["--credentials-file", str(credentials)]) == 0
    calls = entries(fake_ssh)
    assert len(calls) == 2 and "-O" in calls[1]["argv"]
    assert all("helper_path" not in call for call in calls)
    assert "existing lifetime is unchanged" in capsys.readouterr().out


def test_new_master_uses_private_askpass_and_keeps_literal_password(credentials, fake_ssh, capsys, tmp_path):
    assert connector.main(["--credentials-file", str(credentials)]) == 0
    call = entries(fake_ssh)[-1]
    assert "-MNf" in call["argv"] and call["argv"][-1] == connector.ALIAS
    assert "ControlPersist=24h" in options(call["argv"])
    assert "StrictHostKeyChecking=yes" in options(call["argv"])
    assert "BatchMode=no" in options(call["argv"])
    assert call["password_sha256"] == hashlib.sha256((SECRET + "\n").encode()).hexdigest()
    assert call["secret_in_argv_or_env"] is False
    assert call["helper_mode"] == 0o700 and call["directory_mode"] == 0o700
    assert call["isolated_session"] is True and call["askpass_exit"] == 0
    assert not Path(call["helper_path"]).exists()
    output = capsys.readouterr()
    assert SECRET not in output.out + output.err + fake_ssh.read_text()
    assert not (tmp_path / "forbidden").exists()


def test_verify_login_bypasses_master_without_closing_it(credentials, fake_ssh, monkeypatch, capsys):
    monkeypatch.setenv("TEST_REUSE", "1")
    assert connector.main(["--verify-login", "--credentials", str(credentials)]) == 0
    calls = entries(fake_ssh)
    assert len(calls) == 2 and all("-O" not in call["argv"] for call in calls)
    call = calls[-1]
    assert {"ControlPath=none", "ControlMaster=no", "ControlPersist=no"} <= set(options(call["argv"]))
    assert call["argv"][-2:] == [connector.ALIAS, "true"] and "-T" in call["argv"]
    assert "Fresh SSH authentication verified" in capsys.readouterr().out


def test_new_master_returns_while_daemon_keeps_inherited_stdio(credentials, fake_ssh, monkeypatch):
    monkeypatch.setenv("TEST_DAEMON", "1")
    started = time.monotonic()
    assert connector.main(["--credentials-file", str(credentials)]) == 0
    daemon_pid = entries(fake_ssh)[-1]["daemon_pid"]
    try:
        assert time.monotonic() - started < 2
        os.kill(daemon_pid, 0)
    finally:
        os.kill(daemon_pid, signal.SIGTERM)


@pytest.mark.parametrize("mode", [0o644, 0o660, 0o604])
def test_public_credentials_are_rejected_before_ssh(credentials, fake_ssh, mode, capsys):
    credentials.chmod(mode)
    assert connector.main(["--credentials-file", str(credentials)]) == 1
    assert not fake_ssh.exists() and SECRET not in capsys.readouterr().err


def test_readonly_private_credentials_are_valid(credentials):
    credentials.chmod(0o400)
    assert connector.read_credentials(credentials, include_password=True) == (
        "m260012@aspire2antu.nscc.sg", SECRET)


def test_symlink_credentials_are_rejected(credentials, fake_ssh, tmp_path):
    link = tmp_path / "symlink"
    link.symlink_to(credentials)
    assert connector.main(["--credentials-file", str(link)]) == 1
    assert not fake_ssh.exists()


def test_other_owner_credentials_are_rejected(credentials, monkeypatch):
    monkeypatch.setattr(connector.os, "getuid", lambda: os.stat(credentials).st_uid + 1)
    with pytest.raises(ValueError):
        connector.read_credentials(credentials)


@pytest.mark.parametrize("target", [
    "ssh m260012@aspire2antu.nscc.sg; echo forbidden",
    "ssh -oStrictHostKeyChecking=no m260012@aspire2antu.nscc.sg",
    "ssh m260012@elsewhere.example",
    "ssh m260012@aspire2antu.nscc.sg true",
    "ssh m260012@aspire2antu.nscc.sg\nssh m260012@aspire2antu.nscc.sg",
])
def test_only_one_plain_allowed_target_is_accepted(credentials, fake_ssh, target, capsys):
    credentials.write_text(f"Explanation\n{target}\npassword {SECRET}\n")
    assert connector.main(["--credentials-file", str(credentials)]) == 1
    assert not fake_ssh.exists()
    assert target not in capsys.readouterr().err


@pytest.mark.parametrize("prompt", ["Are you sure you want to continue connecting?", "Enter passphrase for key:", "OTP:"])
def test_askpass_refuses_other_prompts(credentials, fake_ssh, monkeypatch, prompt, capsys):
    monkeypatch.setenv("TEST_PROMPT", prompt)
    assert connector.main(["--verify-login", "--credentials-file", str(credentials)]) == 255
    call = entries(fake_ssh)[-1]
    assert call["askpass_exit"] == 1
    assert call["password_sha256"] == hashlib.sha256(b"").hexdigest()
    assert SECRET not in capsys.readouterr().err


def test_authentication_error_does_not_print_subprocess_output(credentials, fake_ssh, monkeypatch, capsys):
    monkeypatch.setenv("TEST_AUTH_EXIT", "255")
    assert connector.main(["--verify-login", "--credentials-file", str(credentials)]) == 255
    output = capsys.readouterr()
    assert "exit 255" in output.err and SECRET not in output.err + output.out


@pytest.mark.parametrize("password_line", ["password ", "password one\npassword two", "missing password"])
def test_missing_or_duplicate_password_is_rejected(credentials, fake_ssh, password_line):
    credentials.write_text(f"ssh m260012@aspire2antu.nscc.sg\n{password_line}\n")
    assert connector.main(["--credentials-file", str(credentials)]) == 1
    assert not fake_ssh.exists()
