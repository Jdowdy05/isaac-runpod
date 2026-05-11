#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DATASET_PATH="${HUMANOID_TELEOP_DATASET_PATH:-${G1_TELEOP_DATASET_PATH:-}}"
if [[ -z "${DATASET_PATH}" ]]; then
  for candidate in \
    "${PROJECT_ROOT}/data/processed/g1/teleop_sparse_pose.npz" \
    "${PROJECT_ROOT}/data/processed/g1/aist_sparse_pose.npz"
  do
    if [[ -f "${candidate}" ]]; then
      DATASET_PATH="${candidate}"
      break
    fi
  done
fi
NUM_ENVS="${NUM_ENVS:-4096}"
source "${PROJECT_ROOT}/scripts/runpod/common.sh"
ISAACLAB_ROOT="$(resolve_isaaclab_root)"
export ISAACLAB_ROOT

if [[ -n "${HUMANOID_TELEOP_MODE:-}" ]]; then
  TELEOP_MODE="${HUMANOID_TELEOP_MODE}"
elif [[ -n "${G1_TELEOP_MODE:-}" ]]; then
  TELEOP_MODE="${G1_TELEOP_MODE}"
elif [[ -n "${DATASET_PATH}" && -f "${DATASET_PATH}" ]]; then
  TELEOP_MODE="dataset"
else
  TELEOP_MODE="synthetic"
fi

ARGS=(
  --task Isaac-G1-Teleop-Direct-v0
  --num_envs "${NUM_ENVS}"
  --teleop_mode "${TELEOP_MODE}"
  --disable_env_add_diff_reward
  --headless
)

if [[ -n "${DATASET_PATH}" && -f "${DATASET_PATH}" ]]; then
  ARGS+=(--teleop_dataset_path "${DATASET_PATH}")
fi

resolve_python_cmd "${ISAACLAB_ROOT}"
"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/rsl_rl/train.py" "${ARGS[@]}" "$@"
