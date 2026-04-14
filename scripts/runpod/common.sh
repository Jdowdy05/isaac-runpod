#!/usr/bin/env bash

resolve_isaaclab_root() {
  if [[ -n "${ISAACLAB_ROOT:-}" ]]; then
    printf '%s\n' "${ISAACLAB_ROOT}"
    return 0
  fi

  if [[ -d "/workspace/isaaclab" ]]; then
    printf '%s\n' "/workspace/isaaclab"
    return 0
  fi

  if [[ -d "/workspace/IsaacLab" ]]; then
    printf '%s\n' "/workspace/IsaacLab"
    return 0
  fi

  printf '%s\n' "/workspace/isaaclab"
}

resolve_python_cmd() {
  local isaaclab_root="${1:-}"

  if [[ -n "${PYTHON_BIN:-}" ]]; then
    PYTHON_CMD=("${PYTHON_BIN}")
    return 0
  fi

  if [[ -n "${isaaclab_root}" && -x "${isaaclab_root}/isaaclab.sh" ]]; then
    PYTHON_CMD=("${isaaclab_root}/isaaclab.sh" -p)
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=("python")
    return 0
  fi

  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=("python3")
    return 0
  fi

  echo "No usable Python interpreter found. Set PYTHON_BIN or ensure Isaac Lab is installed at ISAACLAB_ROOT." >&2
  return 1
}
