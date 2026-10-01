#!/bin/bash
# Source this file: source scripts/activate.sh [lingbot-vla|robotwin-sim]
LRVLA_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export LRVLA_ROOT
export XDG_CACHE_HOME="${LRVLA_ROOT}/.cache"
export HF_HOME="${XDG_CACHE_HOME}/huggingface"
export HF_LEROBOT_HOME="${LRVLA_ROOT}/data/lerobot"
export PIP_CACHE_DIR="${XDG_CACHE_HOME}/pip"
export PYTHONNOUSERSITE=1
export TOKENIZERS_PARALLELISM=false
LRVLA_ENV_NAME="${1:-lingbot-vla}"
if [[ "${LRVLA_ENV_NAME}" == robotwin-sim ]]; then
    export PYTHONPATH="${LRVLA_ROOT}/src:${LRVLA_ROOT}/vendor/RoboTwin:${LRVLA_ROOT}/vendor/RoboTwin/description/utils"
else
    export PYTHONPATH="${LRVLA_ROOT}/src:${LRVLA_ROOT}/vendor/lingbot-vla-v2${PYTHONPATH:+:${PYTHONPATH}}"
fi
export QWEN3VL_PATH="${QWEN3VL_PATH:-${LRVLA_ROOT}/models/Qwen3-VL-4B-Instruct}"
export QWEN3_PATH="${QWEN3VL_PATH}"
export CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
export CONDA_ENVS_PATH="${CONDA_ROOT}/envs"
# Compiler activation hooks assume optional variables may be unset.
LRVLA_RESTORE_NOUNSET=0
[[ $- == *u* ]] && LRVLA_RESTORE_NOUNSET=1
set +u
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
conda activate "${LRVLA_ENV_NAME}"
if [[ "${LRVLA_ENV_NAME}" == robotwin-sim ]]; then
    export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi
[[ "${LRVLA_RESTORE_NOUNSET}" == 1 ]] && set -u
unset LRVLA_RESTORE_NOUNSET
"${CONDA_PREFIX}/bin/python" "${LRVLA_ROOT}/scripts/check_wsl_tools.py" \
    "python=${CONDA_PREFIX}/bin/python" git=/usr/bin/git curl=/usr/bin/curl >/dev/null || return 1
unset LRVLA_ENV_NAME
