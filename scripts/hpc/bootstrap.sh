#!/bin/bash
# Run inside a PBS compute-node allocation, not on an HPC login node.
set -euo pipefail

usage() {
    cat <<'EOF'
Usage: bash scripts/hpc/bootstrap.sh

Run this installer inside a PBS compute-node job (PBS_JOBID must be set).
It installs Miniconda at $HOME/miniconda3, creates the lingbot-vla environment,
and installs the same pinned sources and dependencies as scripts/bootstrap.sh.
It uses Conda's Python instead of the older system Python on RHEL 8 and leaves
shell profiles unchanged: no sudo, apt, or conda init is used.

Environment:
  CONDA_ROOT          Existing or new Linux Conda prefix (default: ~/miniconda3)
  INSTALL_FLASH_ATTN  0 to skip, 1 to install the pinned wheel (default: 0)
  INSTALL_RERUN       Optional dataset viewer, requires glibc >= 2.31 (default: 0)
  BOOTSTRAP_GIT       Explicit Linux Git executable; otherwise select from PATH
  BOOTSTRAP_GIT_MODULE  Module to load if Git is missing (default: git/2.39.2)

After installation: source scripts/activate.sh
EOF
}

if [[ "${1:-}" == --help || "${1:-}" == -h ]]; then
    usage
    exit 0
fi
if [[ "$#" != 0 ]]; then
    usage >&2
    exit 2
fi
if [[ -z "${PBS_JOBID:-}" ]]; then
    printf 'HPC installation must run inside a PBS compute-node allocation; PBS_JOBID is unset.\n' >&2
    printf 'Submit this script through the scheduler rather than running it on a login node.\n' >&2
    exit 2
fi
if [[ "$(/usr/bin/uname -s)" != Linux || "$(/usr/bin/uname -m)" != x86_64 ]]; then
    printf 'The pinned Miniconda installer requires Linux x86_64.\n' >&2
    exit 2
fi

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${PROJECT_DIR}"
export CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
export CONDA_ENVS_PATH="${CONDA_ROOT}/envs"
export PYTHONNOUSERSITE=1
mkdir -p bootstrap

# Keep these identical to scripts/bootstrap.sh: verified snapshot, 2026-10-01.
INSTALLER="Miniconda3-py314_26.7.1-1-Linux-x86_64.sh"
INSTALLER_SHA="e8b25b92b262499141c5bd57a98d3c008024185fa951494b9cd9b6d94e72338b"
if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    [[ -f "bootstrap/${INSTALLER}" ]] || /usr/bin/curl -fL --retry 3 \
        "https://repo.anaconda.com/miniconda/${INSTALLER}" -o "bootstrap/${INSTALLER}"
    printf '%s  %s\n' "${INSTALLER_SHA}" "bootstrap/${INSTALLER}" | /usr/bin/sha256sum -c -
    if [[ -d "${CONDA_ROOT}" ]]; then
        /bin/bash "bootstrap/${INSTALLER}" -b -u -p "${CONDA_ROOT}"
    else
        /bin/bash "bootstrap/${INSTALLER}" -b -p "${CONDA_ROOT}"
    fi
fi
if [[ ! -x "${CONDA_ROOT}/bin/python" ]]; then
    printf 'Miniconda Python is missing from %s/bin/python.\n' "${CONDA_ROOT}" >&2
    exit 1
fi
"${CONDA_ROOT}/bin/python" - <<'PY'
import sys
if sys.platform != 'linux' or sys.version_info < (3, 10):
    raise SystemExit('The bootstrap helpers require Linux Python 3.10 or newer.')
print('Bootstrap Python:', sys.executable, sys.version.split()[0])
PY

if [[ -z "${BOOTSTRAP_GIT:-}" ]]; then
    BOOTSTRAP_GIT="$(command -v git || true)"
    if [[ -z "${BOOTSTRAP_GIT}" ]]; then
        # NSCC compute-node images lack /usr/bin/git; use the shared Git module.
        if ! type module >/dev/null 2>&1; then
            if [[ ! -r /usr/share/Modules/init/bash ]]; then
                printf 'Git is missing. Set BOOTSTRAP_GIT to a Linux executable or initialize the Git module.\n' >&2
                exit 1
            fi
            set +u
            source /usr/share/Modules/init/bash
            set -u
        fi
        module load "${BOOTSTRAP_GIT_MODULE:-git/2.39.2}"
        BOOTSTRAP_GIT="$(command -v git)"
    fi
fi
export BOOTSTRAP_GIT
export BOOTSTRAP_PYTHON="${CONDA_ROOT}/bin/python"
export BOOTSTRAP_INIT_BASH=0
export INSTALL_FLASH_ATTN="${INSTALL_FLASH_ATTN:-0}"
export INSTALL_RERUN="${INSTALL_RERUN:-0}"
exec /bin/bash "${PROJECT_DIR}/scripts/bootstrap.sh"
