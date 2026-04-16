#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
ISAACLAB_ROOT_DEFAULT="/workspace/isaaclab"
ISAACLAB_ROOT="${ISAACLAB_ROOT:-$ISAACLAB_ROOT_DEFAULT}"

if [[ ! -x "${ISAACLAB_ROOT}/isaaclab.sh" ]]; then
  echo "Could not find isaaclab.sh under ISAACLAB_ROOT=${ISAACLAB_ROOT}" >&2
  exit 1
fi

cd "${PROJECT_ROOT}"
"${ISAACLAB_ROOT}/isaaclab.sh" -p "${PROJECT_ROOT}/scripts/debug/test_newton_ground_contact.py" "$@"
