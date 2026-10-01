#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
mkdir -p bootstrap vendor artifacts .cache/pip
export PIP_CACHE_DIR="${PROJECT_DIR}/.cache/pip"
export PYTHONNOUSERSITE=1
export CONDA_ROOT="${CONDA_ROOT:-${HOME}/miniconda3}"
export CONDA_ENVS_PATH="${CONDA_ROOT}/envs"
/usr/bin/python3 scripts/check_wsl_tools.py bash=/bin/bash git=/usr/bin/git \
    curl=/usr/bin/curl sha256sum=/usr/bin/sha256sum >/dev/null
INSTALLER="Miniconda3-py314_26.7.1-1-Linux-x86_64.sh"
# Snapshot verified against https://repo.anaconda.com/miniconda/ on 2026-10-01.
INSTALLER_SHA="e8b25b92b262499141c5bd57a98d3c008024185fa951494b9cd9b6d94e72338b"
if [[ ! -x "${CONDA_ROOT}/bin/conda" ]]; then
    [[ -f "bootstrap/${INSTALLER}" ]] || /usr/bin/curl -fL --retry 3 "https://repo.anaconda.com/miniconda/${INSTALLER}" -o "bootstrap/${INSTALLER}"
    printf '%s  %s\n' "${INSTALLER_SHA}" "bootstrap/${INSTALLER}" | /usr/bin/sha256sum -c -
    if [[ -d "${CONDA_ROOT}" ]]; then
        /bin/bash "bootstrap/${INSTALLER}" -b -u -p "${CONDA_ROOT}"
    else
        /bin/bash "bootstrap/${INSTALLER}" -b -p "${CONDA_ROOT}"
    fi
fi
/usr/bin/python3 scripts/check_wsl_tools.py "conda=${CONDA_ROOT}/bin/conda" >/dev/null
"${CONDA_ROOT}/bin/conda" init bash
source "${CONDA_ROOT}/etc/profile.d/conda.sh"
if [[ ! -x "${CONDA_ROOT}/envs/lingbot-vla/bin/python" ]]; then
    conda create --override-channels -c conda-forge -n lingbot-vla python=3.12 pip 'ffmpeg<8' -y
fi
conda activate lingbot-vla
"${CONDA_PREFIX}/bin/python" scripts/check_wsl_tools.py "python=${CONDA_PREFIX}/bin/python" >/dev/null
if [[ -d "${PROJECT_DIR}/.cache/wheels/cuda" ]]; then
    export PIP_FIND_LINKS="${PROJECT_DIR}/.cache/wheels/cuda"
    # Pip may prefer an index URL over an identical --find-links wheel.
    # Install verified CUDA wheels explicitly to avoid downloading them twice.
    CUDA_WHEELS=("${PROJECT_DIR}"/.cache/wheels/cuda/*.whl)
    if [[ -f "${CUDA_WHEELS[0]}" ]]; then
        "${CONDA_PREFIX}/bin/python" -m pip install --no-index --no-deps "${CUDA_WHEELS[@]}"
    fi
fi
VLA_REV=be969b8fd117fb70550c5d4bf4bc328211b5b1b6
ROBOTWIN_REV=13c3c47ff4312dd62484bcd51be034af55c062d1
prepare_upstream_source() {
    local repository="$1" remote="$2" revision="$3"
    if [[ -e "${repository}/.git" ]]; then
        /usr/bin/git -C "$repository" cat-file -e "${revision}^{commit}" || /usr/bin/git -C "$repository" fetch --depth 1 origin "$revision"
        /usr/bin/git -C "$repository" checkout --detach "$revision"
    elif [[ -f "${repository}/.lrvla_snapshot.json" ]]; then
        /usr/bin/python3 scripts/verify_vendor_snapshot.py "$repository" "$revision"
    elif [[ -d "$repository" && -n "$(/usr/bin/find "$repository" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
        printf 'Existing vendor directory has no Git metadata or verified snapshot: %s. Use an intact submitted snapshot or a fresh pinned clone.\n' "$repository" >&2
        return 1
    else
        /usr/bin/git clone --depth 1 "$remote" "$repository"
        /usr/bin/git -C "$repository" cat-file -e "${revision}^{commit}" || /usr/bin/git -C "$repository" fetch --depth 1 origin "$revision"
        /usr/bin/git -C "$repository" checkout --detach "$revision"
    fi
}
prepare_upstream_source vendor/lingbot-vla-v2 https://github.com/Robbyant/lingbot-vla-v2.git "$VLA_REV"
prepare_upstream_source vendor/RoboTwin https://github.com/RoboTwin-Platform/RoboTwin.git "$ROBOTWIN_REV"
"${CONDA_PREFIX}/bin/python" -m pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://pypi.org/simple
"${CONDA_PREFIX}/bin/python" -m pip install -r vendor/lingbot-vla-v2/requirements.txt -r requirements-extra.txt
"${CONDA_PREFIX}/bin/python" -m pip install numpydantic==1.9.0 --no-deps
"${CONDA_PREFIX}/bin/python" -m pip install --no-deps 'lerobot @ https://github.com/huggingface/lerobot/archive/refs/tags/v0.4.2.tar.gz'
"${CONDA_PREFIX}/bin/python" -m pip install --no-deps -e vendor/lingbot-vla-v2
"${CONDA_PREFIX}/bin/python" -m pip install --no-deps -e vendor/lingbot-vla-v2/lingbotvla/models/vla/vision_models/MoGe
"${CONDA_PREFIX}/bin/python" -m pip install --no-deps -e vendor/lingbot-vla-v2/lingbotvla/models/vla/vision_models/lingbot-depth
"${CONDA_PREFIX}/bin/python" - <<'PY'
import site
from pathlib import Path
root = Path.cwd()
pth = Path(site.getsitepackages()[0]) / 'lrvla_local.pth'
pth.write_text(str(root / 'src') + '\n')
PY
if [[ "${INSTALL_FLASH_ATTN:-1}" == 1 ]]; then
    FLASH_WHEEL='flash_attn-2.8.3+cu12torch2.8cxx11abiTRUE-cp312-cp312-linux_x86_64.whl'
    [[ -f "bootstrap/${FLASH_WHEEL}" ]] || /usr/bin/curl -fL --retry 3 "https://github.com/Dao-AILab/flash-attention/releases/download/v2.8.3/${FLASH_WHEEL}" -o "bootstrap/${FLASH_WHEEL}"
    "${CONDA_PREFIX}/bin/python" -m pip install --no-deps "bootstrap/${FLASH_WHEEL}"
fi
"${CONDA_PREFIX}/bin/python" -m pip freeze > artifacts/requirements-installed.txt
conda list --explicit > artifacts/conda-lingbot-vla-explicit.txt
"${CONDA_PREFIX}/bin/python" - <<'PY'
import torch
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
print('Torch', torch.__version__, 'CUDA runtime', torch.version.cuda, 'GPU visible', torch.cuda.is_available())
PY
printf 'Activate with: source %s/scripts/activate.sh\n' "${PROJECT_DIR}"
