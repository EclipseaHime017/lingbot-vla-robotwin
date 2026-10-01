#!/bin/bash
# Install into the Linux Conda simulation environment; run from any directory.
set -euo pipefail
project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
conda_root="${CONDA_ROOT:-${HOME}/miniconda3}"
conda_bin="${conda_root}/bin/conda"
sim_prefix="${conda_root}/envs/robotwin-sim"
sim_python="${sim_prefix}/bin/python"
# Use Linux executables directly while preserving the user's environment.
/usr/bin/python3 "${project_root}/scripts/check_wsl_tools.py" \
  "conda=${conda_bin}" git=/usr/bin/git bash=/bin/bash >/dev/null
with_curobo=false
skip_torch=false
for arg in "$@"; do
  case "$arg" in
    --with-curobo) with_curobo=true ;;
    --skip-torch) skip_torch=true ;;
    -h|--help)
      printf '%s\n' 'Usage: bash scripts/setup_sim.sh [--with-curobo] [--skip-torch]' \
        'CuRobo requires CUDA_HOME with nvcc plus a working C++ compiler.'
      exit 0 ;;
    *) printf 'Unknown option: %s\n' "$arg" >&2; exit 2 ;;
  esac
done
mkdir -p "${project_root}/artifacts" "${project_root}/.cache/pip" "${project_root}/.cache/conda"
export PIP_CACHE_DIR="${project_root}/.cache/pip"
export CONDA_PKGS_DIRS="${project_root}/.cache/conda"
# Reuse verified downloads shared with bootstrap when they are already present.
for wheel_dir in "${project_root}/.cache/wheels/cuda" "${project_root}/.cache/wheels/sim"; do
  if [[ -d "$wheel_dir" ]]; then
    export PIP_FIND_LINKS="${wheel_dir}${PIP_FIND_LINKS:+ ${PIP_FIND_LINKS}}"
  fi
done
export PYTHONNOUSERSITE=1
export MPLBACKEND=Agg
export MAX_JOBS="${MAX_JOBS:-2}"
if [[ ! -x "$sim_python" ]]; then
  [[ -x "$conda_bin" ]] || { printf 'Run scripts/bootstrap.sh first.\n' >&2; exit 1; }
  "$conda_bin" create -p "$sim_prefix" -y --override-channels -c conda-forge python=3.10 pip 'ffmpeg<8'
fi
# toppra 0.6.3 is source-only on PyPI, so even the basic sim needs a compiler.
if [[ ! -x /usr/bin/g++ && ! -x "${sim_prefix}/bin/x86_64-conda-linux-gnu-c++" ]]; then
  "$conda_bin" install -p "$sim_prefix" -y --override-channels -c conda-forge 'gxx_linux-64=12'
fi
if "$with_curobo" && [[ ! -x "${sim_prefix}/bin/nvcc" ]]; then
  "$conda_bin" install -p "$sim_prefix" -y --override-channels -c nvidia -c conda-forge \
    'cuda-nvcc=12.8' 'cuda-cudart-dev=12.8' 'cuda-cccl=12.8' 'ninja=1.13'
fi
# Conda compiler activation hooks inspect optional unset shell variables.
set +u
source "${project_root}/scripts/activate.sh" robotwin-sim
set -u
# Training source metadata would otherwise leak into pip's simulation checks.
export PYTHONPATH="${project_root}/vendor/RoboTwin"
export LD_LIBRARY_PATH="${sim_prefix}/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
if ! "$skip_torch"; then
  # --find-links may still select the index URL for the same wheel/version.
  # Install the complete verified CUDA cache explicitly to avoid re-downloading.
  shopt -s nullglob
  cuda_wheels=("${project_root}/.cache/wheels/cuda"/nvidia*.whl)
  shopt -u nullglob
  if [[ "${#cuda_wheels[@]}" -eq 14 ]]; then
    "$sim_python" -m pip install --no-index --no-deps "${cuda_wheels[@]}"
  fi
  # Upstream RoboTwin pins 2.4.1/cu121. Blackwell needs this newer local stack.
  # Linux PyPI 2.8.0 wheels depend on the same CUDA 12.8 NVIDIA libraries.
  "$sim_python" -m pip install torch==2.8.0 torchvision==0.23.0 --index-url https://pypi.org/simple
fi
"$sim_python" - <<'PY'
import sys, torch
assert sys.version_info[:2] == (3, 10), sys.version
assert torch.__version__.split('+')[0] == '2.8.0', torch.__version__
assert torch.version.cuda == '12.8', torch.version.cuda
print('Verified Python 3.10, torch 2.8.0, CUDA runtime 12.8')
PY
# SAPIEN depends on toppra, so prepare its native wheel before the bulk install.
# Build against our NumPy 1.26 ABI instead of isolated-build NumPy 2.x.
"$sim_python" -m pip install numpy==1.26.4 scipy==1.10.1 Cython==3.0.12 setuptools==69.5.1 wheel==0.45.1
"$sim_python" -m pip install toppra==0.6.3 --no-build-isolation
"$sim_python" -m pip install \
  numpy==1.26.4 scipy==1.10.1 sapien==3.0.0b1 mplib==0.2.1 \
  gymnasium==0.29.1 transforms3d==0.4.2 open3d==0.18.0 \
  trimesh==4.4.3 imageio==2.34.2 pyyaml==6.0.2 h5py==3.14.0 \
  opencv-python==4.11.0.86 matplotlib==3.10.6 pillow==11.3.0 \
  av==15.0.0 ffmpeg-python==0.2.0 zarr==2.18.3 numcodecs==0.13.1 pydantic==2.11.7 \
  websockets==15.0.1 msgpack==1.1.1 typing_extensions==4.16.0 \
  huggingface_hub==0.34.4 termcolor==3.1.0 tqdm==4.67.1 psutil==7.0.0 \
  setuptools==69.5.1 setuptools_scm==8.3.1 wheel==0.45.1 ninja==1.13.0 Cython==3.0.12

# Apply the two environment-local compatibility adjustments in RoboTwin's
# pinned install script, with exact matches and idempotent verification.
"$sim_python" - <<'PY'
from pathlib import Path
import mplib
import sapien
planner = Path(mplib.__file__).parent / 'planner.py'
before = 'if np.linalg.norm(delta_twist) < 1e-4 or collide or not within_joint_limit:'
after = 'if np.linalg.norm(delta_twist) < 1e-4 or not within_joint_limit:'
text = planner.read_text()
if before in text:
    planner.write_text(text.replace(before, after))
elif after not in text:
    raise RuntimeError('MPLib planner patch no longer matches; inspect pinned dependency')
loader = Path(sapien.__file__).parent / 'wrapper' / 'urdf_loader.py'
text = loader.read_text()
for name in ('urdf_file', 'srdf_file'):
    text = text.replace(f'open({name}, "r")', f'open({name}, "r", encoding="utf-8")')
loader.write_text(text)
print('Applied pinned RoboTwin SAPIEN/MPLib environment patches')
PY

if "$with_curobo"; then
  export CUDA_HOME="${CUDA_HOME:-${sim_prefix}}"
  # NVIDIA's conda toolkit stores headers/libs below targets/, while PyTorch's
  # extension builder also probes CUDA_HOME/include and CUDA_HOME/lib64.
  if [[ -d "${CUDA_HOME}/targets/x86_64-linux/include" ]]; then
    export CPATH="${CUDA_HOME}/targets/x86_64-linux/include${CPATH:+:${CPATH}}"
    export LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LIBRARY_PATH:+:${LIBRARY_PATH}}"
    export LD_LIBRARY_PATH="${CUDA_HOME}/targets/x86_64-linux/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
  fi
  if [[ -z "${CXX:-}" ]]; then
    if [[ -x "${sim_prefix}/bin/x86_64-conda-linux-gnu-c++" ]]; then
      export CXX="${sim_prefix}/bin/x86_64-conda-linux-gnu-c++"
    else
      export CXX=/usr/bin/g++
    fi
  elif [[ "${CXX}" != /* ]]; then
    export CXX="$(command -v "${CXX}")"
  fi
  [[ -x "${CUDA_HOME}/bin/nvcc" && -n "$CXX" ]] || {
    printf 'CuRobo needs nvcc and a C++ compiler in the simulation environment. Install cuda-toolkit 12.8 and GCC/G++ 12, then rerun setup_sim.sh --with-curobo.\n' >&2; exit 1;
  }
  "$sim_python" "${project_root}/scripts/check_wsl_tools.py" \
    "nvcc=${CUDA_HOME}/bin/nvcc" "cxx=${CXX}" >/dev/null
  nvcc_version="$("${CUDA_HOME}/bin/nvcc" --version)"
  [[ "$nvcc_version" == *'release 12.8'* ]] || {
    printf 'CuRobo requires CUDA 12.8 nvcc for this pinned Torch/Blackwell stack.\n' >&2; exit 1;
  }
  export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$("$sim_python" -c 'import torch; c=torch.cuda.get_device_capability(); print(f"{c[0]}.{c[1]}")')}"
  curobo_dir="${project_root}/vendor/RoboTwin/envs/curobo"
  if [[ ! -d "$curobo_dir/.git" ]]; then
    /usr/bin/git clone --branch v0.7.8 --depth 1 https://github.com/NVlabs/curobo.git "$curobo_dir"
  fi
  [[ "$(/usr/bin/git -C "$curobo_dir" describe --tags --exact-match)" == v0.7.8 ]] || {
    printf 'Existing CuRobo checkout is not v0.7.8.\n' >&2; exit 1;
  }
  "$sim_python" -m pip install warp-lang==1.12.0 yourdfpy==0.0.58 usd-core==24.11 lxml==5.4.0 \
    pybind11==2.13.6 numpy-quaternion==2024.0.7 importlib_resources==6.5.2 scikit-image==0.24.0
  "$sim_python" -m pip install -e "$curobo_dir" --no-build-isolation --no-deps
fi
if [[ -d "${project_root}/vendor/RoboTwin/assets/embodiments" ]]; then
  (
    cd "${project_root}/vendor/RoboTwin"
    "$sim_python" script/update_embodiment_config_path.py
  )
fi
"$sim_python" -m pip check
"$sim_python" -m pip freeze > "${project_root}/artifacts/robotwin-sim-pip-freeze.txt"
"$conda_bin" list -p "$sim_prefix" --explicit > "${project_root}/artifacts/robotwin-sim-conda-explicit.txt"
"$sim_python" "${project_root}/scripts/doctor.py" --scope sim --output "${project_root}/artifacts/doctor-sim.json"
