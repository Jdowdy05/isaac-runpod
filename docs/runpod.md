# RunPod Notes

## Expected Flow

1. Provision the pod.
2. Upload or clone this repository into the pod workspace.
3. Run `scripts/runpod/bootstrap.sh`.
   If you are already inside the pre-built Isaac Lab Docker container, the bootstrap script now skips cloning and reinstalling Isaac Lab and only installs the project package and Python-side extras.
4. Run `scripts/runpod/download_open_datasets.sh`.
5. Set `OP3_CFG_IMPORT` after your OP3 asset config exists.
6. Train with either the Newton or PhysX task id.
7. For the paper-aligned ADD trainer, use `scripts/runpod/train_add_newton.sh` or `scripts/runpod/train_add_physx.sh`.

## Useful Environment Variables

- `ISAACLAB_ROOT`: path to the Isaac Lab checkout.
- `ISAACLAB_REF`: branch, tag, or commit to use.
- `INSTALL_MODE`: `newton` or `physx`.
- `ISAACSIM_PATH`: required for the PhysX path if you have an Isaac Sim binary install.
- `PYTHON_BIN`: optional explicit Python executable. If omitted, the RunPod scripts will use `ISAACLAB_ROOT/isaaclab.sh -p` when available, then fall back to `python` or `python3`.
- `OP3_CFG_IMPORT`: import target for your final OP3 asset config.
- `OP3_TELEOP_MODE`: `synthetic` or `dataset`.
- `OP3_TELEOP_DATASET_PATH`: path to the processed sparse-pose dataset NPZ.
