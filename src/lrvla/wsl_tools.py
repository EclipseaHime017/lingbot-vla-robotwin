"""Select Linux executables without changing the user's inherited environment."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import shutil
import sys


def _check_executable(path: Path, seen: set[Path]) -> Path:
    path = path.expanduser().resolve(strict=True)
    if path in seen:
        raise ValueError(f"Executable interpreter cycle: {path}")
    seen.add(path)
    if not path.is_file() or not os.access(path, os.X_OK):
        raise ValueError(f"Not an executable file: {path}")
    with path.open("rb") as stream:
        header = stream.read(4096)
    if header.startswith(b"\x7fELF"):
        return path
    if header.startswith(b"#!"):
        interpreter = shlex.split(header.splitlines()[0][2:].decode("utf-8"))
        if not interpreter or not Path(interpreter[0]).is_absolute():
            raise ValueError(f"Script needs an absolute Linux interpreter: {path}")
        _check_executable(Path(interpreter[0]), seen)
        if Path(interpreter[0]).name == "env":
            arguments = interpreter[1:]
            if arguments[:1] == ["-S"]:
                arguments = arguments[1:]
            if not arguments or arguments[0].startswith("-"):
                raise ValueError(f"Cannot verify env interpreter: {path}")
            target = shutil.which(arguments[0])
            if not target:
                raise FileNotFoundError(arguments[0])
            _check_executable(Path(target), seen)
        return path
    raise ValueError(f"Expected a Linux ELF executable or Linux-interpreted script: {path}")


def linux_executable(name: str, preferred: str | Path | None = None) -> Path:
    """Verify a specific Conda executable or select a known Linux system tool."""
    if sys.platform != "linux":
        raise RuntimeError("Run this project inside WSL/Linux")
    if preferred is not None:
        candidate = Path(preferred)
    elif (Path("/usr/bin") / name).is_file():
        candidate = Path("/usr/bin") / name
    else:
        found = shutil.which(name)
        if not found:
            raise FileNotFoundError(name)
        candidate = Path(found)
    return _check_executable(candidate, set())
