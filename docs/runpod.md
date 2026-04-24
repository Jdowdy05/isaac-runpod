# RunPod Notes

## Expected Flow

1. Provision the pod.
2. Upload or clone this repository into the pod workspace.
3. Run `scripts/runpod/bootstrap.sh`.
   If you are already inside the pre-built Isaac Lab Docker container, the bootstrap script now skips cloning and reinstalling Isaac Lab and only installs the project package and Python-side extras.
4. Run `scripts/runpod/download_open_datasets.sh`.
5. Set `OP3_CFG_IMPORT` after your OP3 asset config exists.
6. Train with the PhysX task id unless you are explicitly debugging Newton contact behavior.
7. For the paper-aligned standalone ADD trainer, use `scripts/runpod/train_add_physx.sh`.
8. For stock RSL-RL PPO with the same pelvis-frame sparse-pose task and dense ADD-style environment reward, use `scripts/runpod/train_rsl_physx.sh`.
9. For RSL-RL PPO plus the true online ADD adversarial discriminator, use `scripts/runpod/train_rsl_add_physx.sh`.
10. Treat Newton scripts as experimental until the OP3 Newton ground-contact issue is resolved.

## Useful Environment Variables

- `ISAACLAB_ROOT`: path to the Isaac Lab checkout. If unset, the scripts prefer `/workspace/isaaclab`, then `/workspace/IsaacLab`.
- `ISAACLAB_REF`: branch, tag, or commit to use.
- `INSTALL_MODE`: `newton` or `physx`.
- `ISAACSIM_PATH`: required for the PhysX path if you have an Isaac Sim binary install.
- `PYTHON_BIN`: optional explicit Python executable. If omitted, the RunPod scripts will use `ISAACLAB_ROOT/isaaclab.sh -p` when available, then fall back to `python` or `python3`.
- `OP3_CFG_IMPORT`: import target for your final OP3 asset config.
- `OP3_TELEOP_MODE`: `synthetic` or `dataset`.
- `OP3_TELEOP_DATASET_PATH`: path to the processed sparse-pose dataset NPZ.
