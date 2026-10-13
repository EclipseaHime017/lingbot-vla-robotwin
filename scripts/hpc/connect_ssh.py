#!/usr/bin/env python3
"""Reuse an NSCC SSH master, or authenticate through a private local file."""
from __future__ import annotations

import argparse
import hashlib
import os
from pathlib import Path
import re
import stat
import subprocess
import sys
import tempfile


ROOT = Path(__file__).resolve().parents[2]
SSH = "/usr/bin/ssh"
ALIAS = "nscc-aspire2a-ntu"
TARGET = re.compile(r"ssh[ \t]+([A-Za-z0-9][A-Za-z0-9_.-]*@aspire2antu\.nscc\.sg)[ \t]*")
CREDENTIALS_ENV = "LRVLA_SSH_CREDENTIALS_FILE"


def read_credentials(path: Path, *, include_password: bool = False) -> tuple[str, str | None]:
    """Check the opened file itself; never interpolate its contents into commands."""
    fd = os.open(path, os.O_RDONLY | os.O_NOFOLLOW)
    with os.fdopen(fd, encoding="utf-8") as stream:
        info = os.fstat(stream.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
            raise ValueError("Credentials must be an owned regular file with private permissions (chmod 600).")
        target, password, password_count = None, None, 0
        for raw_line in stream:
            line = raw_line.rstrip("\r\n")
            if line.startswith("ssh ") or line.startswith("ssh\t"):
                match = TARGET.fullmatch(line)
                if match is None or target is not None:
                    raise ValueError("Credentials must contain one plain SSH target for aspire2antu.nscc.sg.")
                target = match.group(1)
            elif line.startswith("password "):
                password_count += 1
                if include_password:
                    password = line[len("password "):]
                if len(line) == len("password "):
                    raise ValueError("Credentials must contain one nonempty literal password line.")
        if target is None or password_count != 1:
            raise ValueError("Credentials must contain one SSH target and one password line.")
        return target, password


def askpass() -> int:
    # Do not answer host-key, key-passphrase, or unrelated interactive prompts.
    if len(sys.argv) != 2 or not re.search(r"password:\s*$", sys.argv[1], re.IGNORECASE):
        return 1
    try:
        _, password = read_credentials(Path(os.environ[CREDENTIALS_ENV]), include_password=True)
        sys.stdout.write(password + "\n")
        sys.stdout.flush()
        return 0
    except (OSError, ValueError, KeyError, UnicodeError):
        return 1


def connection_profile(target: str) -> tuple[str, Path]:
    """Use the existing alias only if it identifies the file's exact target."""
    result = subprocess.run([SSH, "-G", ALIAS], capture_output=True, text=True, timeout=10, check=False)
    config = dict(line.split(" ", 1) for line in result.stdout.splitlines() if " " in line)
    configured_target = f"{config.get('user', '')}@{config.get('hostname', '')}"
    if result.returncode == 0 and configured_target == target:
        control_path = Path(config.get("controlpath", "none")).expanduser()
        if control_path.is_absolute():
            return ALIAS, control_path
    token = hashlib.sha256(target.encode()).hexdigest()[:32]
    return target, Path.home() / ".ssh/controlmasters" / token


def authenticate(credentials: Path, destination: str, options: list[str]) -> int:
    with tempfile.TemporaryDirectory(prefix="lrvla-ssh-askpass-") as temporary:
        helper = Path(temporary) / "askpass"
        helper.write_text(
            f"#!{sys.executable}\nimport runpy\n"
            f"runpy.run_path({str(Path(__file__).resolve())!r}, run_name='__ssh_askpass__')\n"
        )
        helper.chmod(0o700)
        env = os.environ.copy()
        env.update({"SSH_ASKPASS": str(helper), "SSH_ASKPASS_REQUIRE": "force",
                    "DISPLAY": "lrvla-ssh-askpass", CREDENTIALS_ENV: str(credentials)})
        command = [SSH, "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=no",
                   "-o", "PreferredAuthentications=password,keyboard-interactive",
                   "-o", "PubkeyAuthentication=no", "-o", "NumberOfPasswordPrompts=1",
                   "-o", "ConnectTimeout=15", *options, destination]
        if "-T" in options:
            command.append("true")
        # OpenSSH gets the password from askpass's stdout, never argv or the environment.
        result = subprocess.run(command, env=env, stdin=subprocess.DEVNULL,
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                                start_new_session=True, timeout=45, check=False)
        return result.returncode


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--credentials-file", "--credentials", type=Path, default=ROOT / "sshkey.txt")
    parser.add_argument("--verify-login", action="store_true", help="Authenticate afresh and run remote true.")
    args = parser.parse_args(argv)
    try:
        if sys.platform != "linux":
            raise ValueError("Run this helper in Linux/WSL using Linux OpenSSH.")
        credentials = args.credentials_file.expanduser().absolute()
        target, _ = read_credentials(credentials)
        destination, control_path = connection_profile(target)
        if args.verify_login:
            code = authenticate(credentials, destination, ["-o", "ControlPath=none", "-o", "ControlMaster=no",
                                                           "-o", "ControlPersist=no", "-T"])
            if code == 0:
                print(f"Fresh SSH authentication verified for {target}.")
                return 0
        else:
            check = subprocess.run([SSH, "-o", "StrictHostKeyChecking=yes", "-o", "BatchMode=yes",
                                    "-S", str(control_path), "-O", "check", destination],
                                   capture_output=True, timeout=10, check=False)
            if check.returncode == 0:
                print(f"Reused existing SSH master for {target}; its existing lifetime is unchanged.")
                return 0
            control_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
            directory = control_path.parent.stat()
            if directory.st_uid != os.getuid() or directory.st_mode & 0o077:
                raise ValueError("SSH control directory must be owned by you with private permissions.")
            code = authenticate(credentials, destination, ["-S", str(control_path), "-o", "ControlMaster=yes",
                                                           "-o", "ControlPersist=24h", "-MNf"])
            if code == 0:
                print(f"Created SSH master for {target} with ControlPersist=24h.")
                return 0
        print(f"SSH authentication failed (exit {code}); check credentials and the existing known_hosts entry.",
              file=sys.stderr)
        return code if code > 0 else 1
    except (OSError, ValueError, UnicodeError, subprocess.TimeoutExpired):
        # Do not echo file contents, prompts, subprocess output, or exceptions containing them.
        print("SSH connection failed; check the private credentials file, Linux SSH, and known_hosts.", file=sys.stderr)
        return 1


if __name__ == "__ssh_askpass__":
    raise SystemExit(askpass())
if __name__ == "__main__":
    raise SystemExit(main())
