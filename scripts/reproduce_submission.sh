#!/bin/bash
# One-click reproduction of the submitted checkpoint (expert-filter, 6,000 updates) on one node.
#
#   stage 1: five-epoch full Action Expert BC from the base model (configs/train_expert_full_5epochs.yaml)
#   stage 2: warm start on screened demonstrations (configs/train_expert_full_filter.yaml)
#
# Requires the environments of README section 2 and NPROC GPUs (default 4, A100-40GB class).
# Each step skips work that is already complete where the underlying script supports it.
# Result: runs/expert-filter/checkpoints/global_step_6000/hf_ckpt (FP32 export).
set -euo pipefail
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${PROJECT_DIR}"
source scripts/activate.sh lingbot-vla
NPROC="${NPROC:-4}"
run() { printf '\n===== %s %s\n' "$(date '+%F %T')" "$*"; "$@"; }

# Official assets: base model, Qwen3-VL processor, 50 x 50 clean Aloha-AgileX demonstrations.
run python scripts/download_assets.py --asset base
run python scripts/download_assets.py --asset qwen
run python scripts/download_assets.py --asset clean

# Stage 1 data: LeRobot conversion, episodes 0-44 train / 45-49 validation, train-only statistics.
run python scripts/prepare_data.py --tasks all
run python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    scripts/train.py --config configs/train_expert_full_5epochs.yaml

# Stage 2 data: seen-instruction sidecar, all 50 episodes, statistics-only screening.
run python scripts/prepare_seen_instructions.py
run python scripts/prepare_data.py --tasks all --audit-only --val-episodes 0 \
    --manifest data/clean_manifest_all50.json --norm-output data/clean_norm_stats_all50.json
run python scripts/filter_demos.py scan --output artifacts/demo_filter_reproduced.json
run python - <<'PY'
import json
committed = json.load(open("configs/demo_filter.json"))
reproduced = json.load(open("artifacts/demo_filter_reproduced.json"))
if committed != reproduced:
    raise SystemExit("Screening differs from configs/demo_filter.json; check the clean archives.")
print("Screening reproduces configs/demo_filter.json:", committed["episodes_excluded"], "episodes excluded")
PY
run python scripts/filter_demos.py manifest --manifest data/clean_manifest_all50.json \
    --output data/clean_manifest_filtered.json --norm-output data/clean_norm_stats_filtered.json
run python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node="${NPROC}" \
    scripts/train_warm.py --config configs/train_expert_full_filter.yaml
printf '\nSubmitted checkpoint: %s\n' "${PROJECT_DIR}/runs/expert-filter/checkpoints/global_step_6000/hf_ckpt"
