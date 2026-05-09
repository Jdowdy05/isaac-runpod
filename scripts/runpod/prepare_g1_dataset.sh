#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/data/raw}"
OPEN_PROCESSED_ROOT="${OPEN_PROCESSED_ROOT:-${PROJECT_ROOT}/data/processed/open}"
G1_PROCESSED_ROOT="${G1_PROCESSED_ROOT:-${PROJECT_ROOT}/data/processed/g1}"
AIST_ROOT="${AIST_ROOT:-${RAW_ROOT}/aistplusplus}"
OPEN_AMASS_DATASET_PATH="${OPEN_AMASS_DATASET_PATH:-${OPEN_PROCESSED_ROOT}/amass_sparse_pose.npz}"
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

if [[ ! -f "${OPEN_AMASS_DATASET_PATH}" ]]; then
  echo "Expected OP3-scaled AMASS sparse dataset at: ${OPEN_AMASS_DATASET_PATH}" >&2
  echo "Run scripts/runpod/prepare_amass_dataset.sh first." >&2
  exit 1
fi

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/prepare_aist_sparse.py" \
  --aist-root "${AIST_ROOT}" \
  --output "${G1_AIST_DATASET_PATH}" \
  --embodiment g1

G1_MERGE_OUTPUT_PATH="${G1_COMBINED_DATASET_PATH%.npz}_unfiltered_merge.npz"

"${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/rescale_sparse_dataset.py" \
  --input "${OPEN_AMASS_DATASET_PATH}" \
  --output "${G1_AMASS_DATASET_PATH}" \
  --source-embodiment op3 \
  --target-embodiment g1

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
