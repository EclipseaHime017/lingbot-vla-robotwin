#!/bin/bash
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_DIR}"
source scripts/activate.sh lingbot-vla
"${CONDA_PREFIX}/bin/python" scripts/evaluate.py "$@"
