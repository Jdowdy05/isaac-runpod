#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_PATH="${OP3_TELEOP_DATASET_PATH:-${PROJECT_ROOT}/data/processed/open/aist_sparse_pose.npz}"
source "${PROJECT_ROOT}/scripts/runpod/common.sh"
ISAACLAB_ROOT="$(resolve_isaaclab_root)"

if [[ -n "${OP3_TELEOP_MODE:-}" ]]; then
  TELEOP_MODE="${OP3_TELEOP_MODE}"
elif [[ -f "${DATASET_PATH}" ]]; then
  TELEOP_MODE="dataset"
else
  TELEOP_MODE="synthetic"
fi

ARGS=(
  --task Isaac-OP3-Teleop-Newton-Direct-v0
  --num_envs "${NUM_ENVS:-2048}"
  --teleop_mode "${TELEOP_MODE}"
  --headless
)

if [[ -f "${DATASET_PATH}" ]]; then
  ARGS+=(--teleop_dataset_path "${DATASET_PATH}")
fi

resolve_python_cmd "${ISAACLAB_ROOT}"
"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/add/train.py" "${ARGS[@]}" "$@"
