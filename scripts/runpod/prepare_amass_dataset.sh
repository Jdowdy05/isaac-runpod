#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/data/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/data/processed/open}"
AMASS_ROOT="${AMASS_ROOT:-${RAW_ROOT}/AMASS_Complete}"
SMPL_MODEL_ROOT="${SMPL_MODEL_ROOT:-${RAW_ROOT}/smplh}"
AIST_DATASET_PATH="${AIST_DATASET_PATH:-${PROCESSED_ROOT}/aist_sparse_pose.npz}"
AMASS_DATASET_PATH="${AMASS_DATASET_PATH:-${PROCESSED_ROOT}/amass_sparse_pose.npz}"
COMBINED_DATASET_PATH="${COMBINED_DATASET_PATH:-${PROCESSED_ROOT}/teleop_sparse_pose.npz}"
AMASS_SUBSETS="${AMASS_SUBSETS:-ACCAD BMLmovi BMLrub CMU EKUT EyesJapanDataset HDM05 HumanEva KIT TotalCapture Transitions DanceDB}"

source "${PROJECT_ROOT}/scripts/runpod/common.sh"
ISAACLAB_ROOT="$(resolve_isaaclab_root)"
resolve_python_cmd "${ISAACLAB_ROOT}"

mkdir -p "${PROCESSED_ROOT}"

read -r -a AMASS_SUBSET_ARRAY <<< "${AMASS_SUBSETS}"

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/prepare_amass_sparse.py" \
  --amass-root "${AMASS_ROOT}" \
  --smpl-model-root "${SMPL_MODEL_ROOT}" \
  --output "${AMASS_DATASET_PATH}" \
  --subsets "${AMASS_SUBSET_ARRAY[@]}" \
  "$@"

if [[ ! -f "${AMASS_DATASET_PATH}" ]]; then
  echo "AMASS sparse dataset was not created: ${AMASS_DATASET_PATH}" >&2
  exit 1
fi

MERGE_INPUTS=("${AMASS_DATASET_PATH}")
if [[ -f "${AIST_DATASET_PATH}" ]]; then
  MERGE_INPUTS=("${AIST_DATASET_PATH}" "${AMASS_DATASET_PATH}")
fi

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/merge_sparse_datasets.py" \
  --inputs "${MERGE_INPUTS[@]}" \
  --output "${COMBINED_DATASET_PATH}"

echo
echo "AMASS preprocessing complete."
echo "AMASS sparse dataset: ${AMASS_DATASET_PATH}"
echo "Combined teleop dataset: ${COMBINED_DATASET_PATH}"
