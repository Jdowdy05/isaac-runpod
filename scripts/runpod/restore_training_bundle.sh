#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
DEST="${DEST:-${PROJECT_ROOT}}"
INCOMING_DIR="${INCOMING_DIR:-${PROJECT_ROOT}/.transfer_incoming}"
ARCHIVE_NAME=""
PARTS=0

usage() {
  cat <<EOF
Usage: $(basename "$0") --archive-name NAME --parts N [options]

Receive one or more runpodctl parts on the pod, reassemble them, and extract
them into the current repo checkout.

Options:
  --archive-name NAME   Final archive name produced by send_training_bundle.sh
  --parts N             Number of archive parts to receive
  --dest PATH           Extraction destination (default: ${DEST})
  --incoming-dir PATH   Temporary download directory (default: ${INCOMING_DIR})
  --help                Show this help
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --archive-name)
      ARCHIVE_NAME="$2"
      shift 2
      ;;
    --parts)
      PARTS="$2"
      shift 2
      ;;
    --dest)
      DEST="$2"
      shift 2
      ;;
    --incoming-dir)
      INCOMING_DIR="$2"
      shift 2
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

if [[ -z "${ARCHIVE_NAME}" || "${PARTS}" -le 0 ]]; then
  usage >&2
  exit 1
fi

if ! command -v runpodctl >/dev/null 2>&1; then
  echo "runpodctl is not installed or not on PATH in this pod." >&2
  exit 1
fi

mkdir -p "${INCOMING_DIR}"
mkdir -p "${DEST}"

echo "Incoming directory: ${INCOMING_DIR}"
echo "Destination: ${DEST}"
echo

for ((i = 1; i <= PARTS; i++)); do
  echo "Waiting for part ${i}/${PARTS}."
  read -r -p "Paste the runpodctl receive code: " RECEIVE_CODE
  (
    cd "${INCOMING_DIR}"
    runpodctl receive "${RECEIVE_CODE}"
  )
done

if [[ "${PARTS}" -eq 1 ]]; then
  ARCHIVE_PATH="${INCOMING_DIR}/${ARCHIVE_NAME}"
  if [[ ! -f "${ARCHIVE_PATH}" ]]; then
    echo "Expected archive not found after receive: ${ARCHIVE_PATH}" >&2
    exit 1
  fi
else
  shopt -s nullglob
  PART_FILES=("${INCOMING_DIR}/${ARCHIVE_NAME}.part."*)
  shopt -u nullglob
  if [[ ${#PART_FILES[@]} -ne ${PARTS} ]]; then
    echo "Expected ${PARTS} part files for ${ARCHIVE_NAME}, found ${#PART_FILES[@]}." >&2
    exit 1
  fi
  ARCHIVE_PATH="${INCOMING_DIR}/${ARCHIVE_NAME}"
  cat "${PART_FILES[@]}" > "${ARCHIVE_PATH}"
fi

echo
echo "Extracting ${ARCHIVE_PATH} to ${DEST}"
tar -xzf "${ARCHIVE_PATH}" -C "${DEST}"
echo "Restore complete."
