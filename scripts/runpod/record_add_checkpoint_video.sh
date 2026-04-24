#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  record_add_checkpoint_video.sh --run RUN_FOLDER --iter ITERATION [options] [-- extra playback args]
  record_add_checkpoint_video.sh RUN_FOLDER ITERATION [options] [-- extra playback args]

Options:
  --run RUN_FOLDER       Checkpoint run folder under checkpoints/add, e.g. 2026-04-21.
  --iter ITERATION       Checkpoint iteration, e.g. 5000 or 005000.
  --steps N              Playback steps to record. Default: 500.
  --output PATH          Output mp4 path. Default: checkpoints/add/videos/RUN/op3_add_camera_playback_ITER.mp4.
  --task TASK            Isaac Lab task. Default: Isaac-OP3-Teleop-Direct-v0.
  --use-teacher          Record the teacher policy instead of the deployment student.
  -h, --help             Show this help.
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/runpod/common.sh"

ISAACLAB_ROOT="$(resolve_isaaclab_root)"
resolve_python_cmd "${ISAACLAB_ROOT}"

RUN_FOLDER=""
ITERATION=""
STEPS="${VIDEO_STEPS:-500}"
TASK="${TASK:-Isaac-OP3-Teleop-Direct-v0}"
OUTPUT_PATH=""
USE_TEACHER=0
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run)
      RUN_FOLDER="$2"
      shift 2
      ;;
    --iter|--iteration)
      ITERATION="$2"
      shift 2
      ;;
    --steps)
      STEPS="$2"
      shift 2
      ;;
    --output)
      OUTPUT_PATH="$2"
      shift 2
      ;;
    --task)
      TASK="$2"
      shift 2
      ;;
    --use-teacher)
      USE_TEACHER=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
    *)
      if [[ -z "${RUN_FOLDER}" ]]; then
        RUN_FOLDER="$1"
      elif [[ -z "${ITERATION}" ]]; then
        ITERATION="$1"
      else
        EXTRA_ARGS+=("$1")
      fi
      shift
      ;;
  esac
done

if [[ -z "${RUN_FOLDER}" || -z "${ITERATION}" ]]; then
  usage >&2
  exit 1
fi

if [[ "${ITERATION}" =~ ^[0-9]+$ ]]; then
  ITER_PAD="$(printf "%06d" "$((10#${ITERATION}))")"
else
  echo "Iteration must be numeric: ${ITERATION}" >&2
  exit 1
fi

CHECKPOINT="${PROJECT_ROOT}/checkpoints/add/${RUN_FOLDER}/add_op3_iter_${ITER_PAD}.pt"
if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${PROJECT_ROOT}/checkpoints/add/videos/${RUN_FOLDER}/op3_add_camera_playback_${ITER_PAD}.mp4"
fi
mkdir -p "$(dirname "${OUTPUT_PATH}")"

ARGS=(
  "${PROJECT_ROOT}/scripts/add/record_camera_playback.py"
  --task "${TASK}"
  --checkpoint "${CHECKPOINT}"
  --headless
  --steps "${STEPS}"
  --output "${OUTPUT_PATH}"
)

if [[ "${USE_TEACHER}" == "1" ]]; then
  ARGS+=(--use_teacher)
fi
ARGS+=("${EXTRA_ARGS[@]}")

echo "Recording checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_PATH}"
"${PYTHON_CMD[@]}" "${ARGS[@]}"
