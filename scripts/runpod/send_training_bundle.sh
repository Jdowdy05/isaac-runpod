#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
TRANSFER_ROOT="${TRANSFER_ROOT:-/tmp/isaac-runpod-transfer}"
RUNPOD_DEST="${RUNPOD_DEST:-/workspace/isaac-runpod}"
CHUNK_SIZE_MB="${CHUNK_SIZE_MB:-1500}"
ARCHIVE_NAME="isaac_runpod_training_$(date +%Y%m%d_%H%M%S).tar.gz"
SEND_NOW=1
INCLUDE_RAW=0

usage() {
  cat <<EOF
Usage: $(basename "$0") [options]

Package local training artifacts for transfer back to RunPod with runpodctl.

Default payload:
  - data/processed/open
  - checkpoints/add
  - source/op3_teleop_lab/op3_teleop_lab/assets/op3_asset

Options:
  --archive-name NAME      Override output archive name
  --chunk-size-mb N        Split archive into N MB parts (default: ${CHUNK_SIZE_MB})
  --transfer-root PATH     Directory to store the packaged archive/parts
  --runpod-dest PATH       Destination repo root on the pod (default: ${RUNPOD_DEST})
  --path RELPATH           Add a repo-relative path to the payload (may be repeated)
  --include-raw            Also include data/raw/AMASS_Complete and data/raw/smplh if present
  --package-only           Only package and split; do not invoke runpodctl send
  --send-now               Explicitly invoke runpodctl send for each part after packaging
  --help                   Show this help
EOF
}

declare -a USER_PATHS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive-name)
      ARCHIVE_NAME="$2"
      shift 2
      ;;
    --chunk-size-mb)
      CHUNK_SIZE_MB="$2"
      shift 2
      ;;
    --transfer-root)
      TRANSFER_ROOT="$2"
      shift 2
      ;;
    --runpod-dest)
      RUNPOD_DEST="$2"
      shift 2
      ;;
    --path)
      USER_PATHS+=("$2")
      shift 2
      ;;
    --include-raw)
      INCLUDE_RAW=1
      shift
      ;;
    --package-only)
      SEND_NOW=0
      shift
      ;;
    --send-now)
      SEND_NOW=1
      shift
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

declare -a DEFAULT_PATHS=(
  "data/processed/open"
  "checkpoints/add"
  "source/op3_teleop_lab/op3_teleop_lab/assets/op3_asset"
)

declare -a PAYLOAD_PATHS=()
if [[ ${#USER_PATHS[@]} -gt 0 ]]; then
  PAYLOAD_PATHS=("${USER_PATHS[@]}")
else
  for rel_path in "${DEFAULT_PATHS[@]}"; do
    if [[ -e "${PROJECT_ROOT}/${rel_path}" ]]; then
      PAYLOAD_PATHS+=("${rel_path}")
    fi
  done
fi

if [[ ${INCLUDE_RAW} -eq 1 ]]; then
  for raw_path in "data/raw/AMASS_Complete" "data/raw/smplh"; do
    if [[ -e "${PROJECT_ROOT}/${raw_path}" ]]; then
      PAYLOAD_PATHS+=("${raw_path}")
    fi
  done
fi

if [[ ${#PAYLOAD_PATHS[@]} -eq 0 ]]; then
  echo "No payload paths were found to package." >&2
  exit 1
fi

for rel_path in "${PAYLOAD_PATHS[@]}"; do
  if [[ ! -e "${PROJECT_ROOT}/${rel_path}" ]]; then
    echo "Missing payload path: ${PROJECT_ROOT}/${rel_path}" >&2
    exit 1
  fi
done

mkdir -p "${TRANSFER_ROOT}"
ARCHIVE_PATH="${TRANSFER_ROOT}/${ARCHIVE_NAME}"

echo "Packaging payload:"
for rel_path in "${PAYLOAD_PATHS[@]}"; do
  echo "  - ${rel_path}"
done
echo

tar -C "${PROJECT_ROOT}" -czf "${ARCHIVE_PATH}" "${PAYLOAD_PATHS[@]}"

PART_PREFIX="${ARCHIVE_PATH}.part."
PART_SIZE_ARG="${CHUNK_SIZE_MB}m"
split -b "${PART_SIZE_ARG}" -d -a 3 "${ARCHIVE_PATH}" "${PART_PREFIX}"

shopt -s nullglob
PART_FILES=("${PART_PREFIX}"*)
shopt -u nullglob

if [[ ${#PART_FILES[@]} -eq 0 ]]; then
  PART_FILES=("${ARCHIVE_PATH}")
fi

MANIFEST_PATH="${TRANSFER_ROOT}/${ARCHIVE_NAME%.tar.gz}.manifest.txt"
{
  echo "archive_name=${ARCHIVE_NAME}"
  echo "parts=${#PART_FILES[@]}"
  echo "runpod_dest=${RUNPOD_DEST}"
  echo "payload_paths=${PAYLOAD_PATHS[*]}"
} > "${MANIFEST_PATH}"

echo "Created:"
for part in "${PART_FILES[@]}"; do
  du -h "${part}"
done
echo "Manifest: ${MANIFEST_PATH}"
echo
echo "Run this on the pod before you start receiving:"
echo "  cd ${RUNPOD_DEST}"
echo "  bash scripts/runpod/restore_training_bundle.sh --archive-name ${ARCHIVE_NAME} --parts ${#PART_FILES[@]}"
echo

if [[ ${SEND_NOW} -eq 0 ]]; then
  echo "To send manually from this machine:"
  for part in "${PART_FILES[@]}"; do
    echo "  runpodctl send ${part}"
  done
  exit 0
fi

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl is not installed or not on PATH." >&2
  exit 1
fi

for idx in "${!PART_FILES[@]}"; do
  part="${PART_FILES[$idx]}"
  echo
  echo "Sending part $((idx + 1))/${#PART_FILES[@]}: ${part}"
  echo "Make sure the pod-side restore script is waiting for the next receive code."
  runpodctl send "${part}"
done

echo
echo "All parts sent. On the pod, let the restore script finish extraction."
