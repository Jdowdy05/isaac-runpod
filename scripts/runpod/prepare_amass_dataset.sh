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
OP3_SPARSE_FILTER_ENABLED="${OP3_SPARSE_FILTER_ENABLED:-1}"
OP3_KEEP_UNFILTERED_MERGE="${OP3_KEEP_UNFILTERED_MERGE:-0}"
OP3_SPARSE_FILTER_ARGS="${OP3_SPARSE_FILTER_ARGS:-}"

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

MERGE_OUTPUT_PATH="${COMBINED_DATASET_PATH}"
if [[ "${OP3_SPARSE_FILTER_ENABLED}" != "0" ]]; then
  MERGE_OUTPUT_PATH="${COMBINED_DATASET_PATH%.npz}_unfiltered_merge.npz"
fi

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/merge_sparse_datasets.py" \
  --inputs "${MERGE_INPUTS[@]}" \
  --output "${MERGE_OUTPUT_PATH}"

if [[ "${OP3_SPARSE_FILTER_ENABLED}" != "0" ]]; then
  read -r -a OP3_SPARSE_FILTER_ARG_ARRAY <<< "${OP3_SPARSE_FILTER_ARGS}"
  "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/filter_sparse_pose_dataset.py" \
    --input "${MERGE_OUTPUT_PATH}" \
    --output "${COMBINED_DATASET_PATH}" \
    "${OP3_SPARSE_FILTER_ARG_ARRAY[@]}"
  if [[ "${OP3_KEEP_UNFILTERED_MERGE}" != "1" ]]; then
    rm -f "${MERGE_OUTPUT_PATH}"
  fi
fi

echo
echo "AMASS preprocessing complete."
echo "AMASS sparse dataset: ${AMASS_DATASET_PATH}"
echo "Combined teleop dataset: ${COMBINED_DATASET_PATH}"
