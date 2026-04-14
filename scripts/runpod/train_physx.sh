#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-/workspace/IsaacLab}"
DATASET_PATH="${OP3_TELEOP_DATASET_PATH:-${PROJECT_ROOT}/data/processed/open/aist_sparse_pose.npz}"
NUM_ENVS="${NUM_ENVS:-1024}"
source "${PROJECT_ROOT}/scripts/runpod/common.sh"

if [[ -n "${OP3_TELEOP_MODE:-}" ]]; then
  TELEOP_MODE="${OP3_TELEOP_MODE}"
elif [[ -f "${DATASET_PATH}" ]]; then
  TELEOP_MODE="dataset"
else
  TELEOP_MODE="synthetic"
fi

export OP3_TELEOP_MODE="${TELEOP_MODE}"
if [[ -f "${DATASET_PATH}" ]]; then
  export OP3_TELEOP_DATASET_PATH="${DATASET_PATH}"
fi

resolve_python_cmd "${ISAACLAB_ROOT}"
"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/rl_games/train.py" \
  --task Isaac-OP3-Teleop-Direct-v0 \
  --headless \
  --num_envs "${NUM_ENVS}" \
  "$@"
