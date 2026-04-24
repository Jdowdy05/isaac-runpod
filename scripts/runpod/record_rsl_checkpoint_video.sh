#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage:
  record_rsl_checkpoint_video.sh --checkpoint PATH [options] [-- extra playback args]

Options:
  --checkpoint PATH     Full path to an RSL or RSL-ADD checkpoint.
  --runner KIND         `ppo` or `add`. Default: ppo.
  --steps N             Playback steps to record. Default: 500.
  --output PATH         Output mp4 path. Default: checkpoint path with .mp4 suffix.
  --task TASK           Isaac Lab task. Default: Isaac-OP3-Teleop-Direct-v0.
  -h, --help            Show this help.
USAGE
}

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source "${PROJECT_ROOT}/scripts/runpod/common.sh"

ISAACLAB_ROOT="$(resolve_isaaclab_root)"
resolve_python_cmd "${ISAACLAB_ROOT}"

CHECKPOINT=""
RUNNER="ppo"
STEPS="${VIDEO_STEPS:-500}"
TASK="${TASK:-Isaac-OP3-Teleop-Direct-v0}"
OUTPUT_PATH=""
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --checkpoint)
      CHECKPOINT="$2"
      shift 2
      ;;
    --runner)
      RUNNER="$2"
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
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -z "${CHECKPOINT}" ]]; then
  usage >&2
  exit 1
fi

if [[ ! -f "${CHECKPOINT}" ]]; then
  echo "Checkpoint does not exist: ${CHECKPOINT}" >&2
  exit 1
fi

if [[ -z "${OUTPUT_PATH}" ]]; then
  OUTPUT_PATH="${CHECKPOINT%.*}.mp4"
fi
mkdir -p "$(dirname "${OUTPUT_PATH}")"

ARGS=(
  "${PROJECT_ROOT}/scripts/rsl_rl/record_camera_playback.py"
  --task "${TASK}"
  --runner "${RUNNER}"
  --checkpoint "${CHECKPOINT}"
  --headless
  --steps "${STEPS}"
  --output "${OUTPUT_PATH}"
)
ARGS+=("${EXTRA_ARGS[@]}")

echo "Recording checkpoint: ${CHECKPOINT}"
echo "Output: ${OUTPUT_PATH}"
"${PYTHON_CMD[@]}" "${ARGS[@]}"
