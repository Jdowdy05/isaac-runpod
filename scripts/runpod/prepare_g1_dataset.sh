#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/data/raw}"
G1_PROCESSED_ROOT="${G1_PROCESSED_ROOT:-${PROJECT_ROOT}/data/processed/g1}"
AIST_ROOT="${AIST_ROOT:-${RAW_ROOT}/aistplusplus}"
AMASS_ROOT="${AMASS_ROOT:-${RAW_ROOT}/AMASS_Complete}"
SMPL_MODEL_ROOT="${SMPL_MODEL_ROOT:-${RAW_ROOT}/smplh}"
AMASS_SUBSETS="${AMASS_SUBSETS:-ACCAD BMLmovi BMLrub CMU EKUT EyesJapanDataset HDM05 HumanEva KIT TotalCapture Transitions DanceDB}"
G1_AIST_DATASET_PATH="${G1_AIST_DATASET_PATH:-${G1_PROCESSED_ROOT}/aist_sparse_pose.npz}"
G1_AMASS_DATASET_PATH="${G1_AMASS_DATASET_PATH:-${G1_PROCESSED_ROOT}/amass_sparse_pose.npz}"
G1_COMBINED_DATASET_PATH="${G1_COMBINED_DATASET_PATH:-${G1_PROCESSED_ROOT}/teleop_sparse_pose.npz}"
G1_KEEP_UNFILTERED_MERGE="${G1_KEEP_UNFILTERED_MERGE:-0}"
G1_SPARSE_FILTER_ARGS="${G1_SPARSE_FILTER_ARGS:-}"

source "${PROJECT_ROOT}/scripts/runpod/common.sh"
ISAACLAB_ROOT="$(resolve_isaaclab_root)"
resolve_python_cmd "${ISAACLAB_ROOT}"

mkdir -p "${G1_PROCESSED_ROOT}"

if [[ ! -d "${AIST_ROOT}" ]]; then
  echo "AIST++ root not found: ${AIST_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${AMASS_ROOT}" ]]; then
  echo "AMASS root not found: ${AMASS_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${SMPL_MODEL_ROOT}" ]]; then
  echo "SMPL-H model root not found: ${SMPL_MODEL_ROOT}" >&2
  exit 1
fi

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/prepare_aist_sparse.py" \
  --aist-root "${AIST_ROOT}" \
  --output "${G1_AIST_DATASET_PATH}" \
  --embodiment g1

G1_MERGE_OUTPUT_PATH="${G1_COMBINED_DATASET_PATH%.npz}_unfiltered_merge.npz"

read -r -a AMASS_SUBSET_ARRAY <<< "${AMASS_SUBSETS}"
"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/prepare_amass_sparse.py" \
  --amass-root "${AMASS_ROOT}" \
  --smpl-model-root "${SMPL_MODEL_ROOT}" \
  --output "${G1_AMASS_DATASET_PATH}" \
  --embodiment g1 \
  --disable-feasibility-filter \
  --subsets "${AMASS_SUBSET_ARRAY[@]}"

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/merge_sparse_datasets.py" \
  --inputs "${G1_AIST_DATASET_PATH}" "${G1_AMASS_DATASET_PATH}" \
  --output "${G1_MERGE_OUTPUT_PATH}"

read -r -a G1_SPARSE_FILTER_ARG_ARRAY <<< "${G1_SPARSE_FILTER_ARGS}"
"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/filter_sparse_pose_dataset.py" \
  --input "${G1_MERGE_OUTPUT_PATH}" \
  --output "${G1_COMBINED_DATASET_PATH}" \
  --embodiment g1 \
  "${G1_SPARSE_FILTER_ARG_ARRAY[@]}"

if [[ "${G1_KEEP_UNFILTERED_MERGE}" != "1" ]]; then
  rm -f "${G1_MERGE_OUTPUT_PATH}"
fi

echo
echo "G1 sparse preprocessing complete."
echo "G1 AIST sparse dataset: ${G1_AIST_DATASET_PATH}"
echo "G1 AMASS sparse dataset: ${G1_AMASS_DATASET_PATH}"
echo "G1 combined teleop dataset: ${G1_COMBINED_DATASET_PATH}"
