#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORKSPACE_ROOT="${WORKSPACE_ROOT:-/workspace}"
source "${PROJECT_ROOT}/scripts/runpod/common.sh"
ISAACLAB_ROOT="$(resolve_isaaclab_root)"
ISAACLAB_REF="${ISAACLAB_REF:-v2.3.2}"
INSTALL_MODE="${INSTALL_MODE:-physx}"

echo "Project root: ${PROJECT_ROOT}"
echo "Isaac Lab root: ${ISAACLAB_ROOT}"
echo "Isaac Lab ref: ${ISAACLAB_REF}"
echo "Install mode: ${INSTALL_MODE}"

mkdir -p "${WORKSPACE_ROOT}"

IS_PREBUILT_DOCKER=0
if [[ -x "${ISAACLAB_ROOT}/isaaclab.sh" && ! -d "${ISAACLAB_ROOT}/.git" ]]; then
  IS_PREBUILT_DOCKER=1
fi

if [[ "${IS_PREBUILT_DOCKER}" == "1" ]]; then
  echo "Detected pre-built Isaac Lab container at ${ISAACLAB_ROOT}; skipping Isaac Lab clone and install."
  echo "WARNING: pre-built containers are not version-pinned by this script. Verify the container actually matches Isaac Lab ${ISAACLAB_REF}."
else
  if [[ ! -d "${ISAACLAB_ROOT}/.git" ]]; then
    git clone https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_ROOT}"
  fi

  git -C "${ISAACLAB_ROOT}" fetch --tags
  git -C "${ISAACLAB_ROOT}" checkout "${ISAACLAB_REF}"
fi

if [[ "${IS_PREBUILT_DOCKER}" == "1" ]]; then
  :
elif [[ "${INSTALL_MODE}" == "physx" ]]; then
  : "${ISAACSIM_PATH:?Set ISAACSIM_PATH to your Isaac Sim binary installation for PhysX mode.}"
  ln -sfn "${ISAACSIM_PATH}" "${ISAACLAB_ROOT}/_isaac_sim"
  (
    cd "${ISAACLAB_ROOT}"
    ./isaaclab.sh --install
  )
else
  echo "Unsupported INSTALL_MODE: ${INSTALL_MODE}. Only physx is supported in the Isaac Lab 2.3.2 migration." >&2
  exit 1
fi

resolve_python_cmd "${ISAACLAB_ROOT}"
"${PYTHON_CMD[@]}" -m pip install --upgrade pip setuptools wheel
"${PYTHON_CMD[@]}" -m pip install --upgrade numpy pyyaml "huggingface_hub[cli]"
"${PYTHON_CMD[@]}" -m pip install -e "${PROJECT_ROOT}/source/op3_teleop_lab"

echo
echo "Bootstrap complete."
echo "Next steps:"
echo "  1. Run scripts/runpod/download_open_datasets.sh"
echo "  2. Add or point OP3_CFG_IMPORT at your final OP3 asset config"
echo "  3. Train with scripts/runpod/train_physx.sh, scripts/runpod/train_add_physx.sh, or scripts/runpod/train_rsl_add_physx.sh"
