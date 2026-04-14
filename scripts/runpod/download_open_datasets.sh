#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RAW_ROOT="${RAW_ROOT:-${PROJECT_ROOT}/data/raw}"
PROCESSED_ROOT="${PROCESSED_ROOT:-${PROJECT_ROOT}/data/processed/open}"
COMBINED_DATASET_PATH="${COMBINED_DATASET_PATH:-${PROCESSED_ROOT}/teleop_sparse_pose.npz}"
DOWNLOAD_AIST="${DOWNLOAD_AIST:-1}"
DOWNLOAD_RETARGETED_AMASS="${DOWNLOAD_RETARGETED_AMASS:-0}"
source "${PROJECT_ROOT}/scripts/runpod/common.sh"
ISAACLAB_ROOT="$(resolve_isaaclab_root)"

AIST_ROOT="${RAW_ROOT}/aistplusplus"
AIST_RELEASE_ROOT="https://github.com/google/aistplusplus_dataset/releases/download/v1.0"

mkdir -p "${RAW_ROOT}" "${PROCESSED_ROOT}"
resolve_python_cmd "${ISAACLAB_ROOT}"

download_file() {
  local url="$1"
  local output_path="$2"
  if [[ -f "${output_path}" ]]; then
    echo "Already downloaded: ${output_path}"
    return
  fi
  curl -L --fail --retry 3 "${url}" -o "${output_path}"
}

if [[ "${DOWNLOAD_AIST}" == "1" ]]; then
  mkdir -p "${AIST_ROOT}"
  download_file "${AIST_RELEASE_ROOT}/motions.zip" "${AIST_ROOT}/motions.zip"
  download_file "${AIST_RELEASE_ROOT}/keypoints3d.zip" "${AIST_ROOT}/keypoints3d.zip"
  download_file "${AIST_RELEASE_ROOT}/cameras.zip" "${AIST_ROOT}/cameras.zip"

  unzip -qo "${AIST_ROOT}/motions.zip" -d "${AIST_ROOT}"
  unzip -qo "${AIST_ROOT}/keypoints3d.zip" -d "${AIST_ROOT}"
  unzip -qo "${AIST_ROOT}/cameras.zip" -d "${AIST_ROOT}"

  "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/prepare_aist_sparse.py" \
    --aist-root "${AIST_ROOT}" \
    --output "${PROCESSED_ROOT}/aist_sparse_pose.npz"

  "${PYTHON_CMD[@]}" "${PROJECT_ROOT}/scripts/data/merge_sparse_datasets.py" \
    --inputs "${PROCESSED_ROOT}/aist_sparse_pose.npz" \
    --output "${COMBINED_DATASET_PATH}"
fi

if [[ "${DOWNLOAD_RETARGETED_AMASS}" == "1" ]]; then
  hf download \
    --repo-type dataset \
    ember-lab-berkeley/AMASS_Retargeted_for_G1 \
    --local-dir "${RAW_ROOT}/hf/AMASS_Retargeted_for_G1"
fi

echo
echo "Open dataset setup complete."
if [[ -f "${COMBINED_DATASET_PATH}" ]]; then
  echo "Teleop sparse dataset: ${COMBINED_DATASET_PATH}"
fi
if [[ "${DOWNLOAD_RETARGETED_AMASS}" == "1" ]]; then
  echo "Retargeted motion prior: ${RAW_ROOT}/hf/AMASS_Retargeted_for_G1"
fi
