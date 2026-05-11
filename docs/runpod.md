# RunPod Notes

## Expected Flow

1. Provision the pod.
2. Upload or clone this repository into the pod workspace.
3. Run `scripts/runpod/bootstrap.sh`.
   If you are already inside the pre-built Isaac Lab Docker container, the bootstrap script now skips cloning and reinstalling Isaac Lab and only installs the project package and Python-side extras.
4. Run `scripts/runpod/download_open_datasets.sh`.
5. Set `OP3_CFG_IMPORT` after your OP3 asset config exists.
6. Train with the PhysX task id. Newton training and diagnostics are disabled in this Isaac Lab `2.3.2` / PhysX-only workflow.

## SSH Access Notes

Current G1 RunPod SSH target:

```bash
ssh x892ggwbr557zd-64410d71@ssh.runpod.io -i ~/.ssh/id_ed25519
```

The RunPod SSH wrapper may fail from a non-interactive/non-PTY client with an error like `Your SSH client doesn't support PTY`. When using Codex or another automation shell, start SSH with an allocated PTY instead of a plain non-interactive pipe. In Codex, run the SSH command through an interactive TTY session (`tty=true`). From a normal shell fallback, force PTY allocation with:

```bash
ssh -tt x892ggwbr557zd-64410d71@ssh.runpod.io -i ~/.ssh/id_ed25519
```

## Idle GPU Utilization Checks

If `nvidia-smi` shows high GPU utilization but no training should be running, check for the pod-template Isaac Sim streaming process before assuming a training job is still alive:

```bash
nvidia-smi
ps -eo pid,ppid,etime,stat,pcpu,pmem,args | egrep 'Isaac|isaac|train|rsl|teacher|python|kit|omni|runheadless'
pgrep -af 'Isaac-G1-Teleop-Direct|Isaac-OP3-Teleop-Direct|scripts/rsl_rl/train|teacher_student|train_add|rl_games'
```

On RunPod `x892ggwbr557zd` as of 2026-05-11, high idle GPU load came from the base container startup process:

```text
/sbin/docker-init -- /isaac-sim/runheadless.sh
/bin/sh /isaac-sim/runheadless.sh
/isaac-sim/kit/kit /isaac-sim/apps/isaacsim.exp.full.streaming.kit --no-window ...
```

That process is Isaac Sim full streaming, not policy training. It can keep the GPU in a high-power P2 state even when no `scripts/rsl_rl/train.py`, `teacher_student`, or ADD job is running.

Do not kill the streaming process if this container was launched with `/isaac-sim/runheadless.sh` as PID 1's child; killing it can make RunPod treat the container as crashed and restart the pod. The current band-aid is to pause it with `SIGSTOP`:

```bash
STREAM_PID="$(pgrep -f 'isaacsim\.exp\.full\.streaming\.kit' | head -n 1)"
kill -STOP "${STREAM_PID}"
nvidia-smi
```

This keeps the container alive and drops active GPU utilization, but it does not free the VRAM already allocated by Isaac Sim. Resume streaming with:

```bash
kill -CONT "${STREAM_PID}"
```

On 2026-05-11, pausing PID `23` on `x892ggwbr557zd` dropped GPU utilization from about `89%` to `0%`; VRAM stayed around `2.8 GB`, as expected.

## OP3 Flow

OP3 uses sparse datasets under `data/processed/open/`.

1. Prepare OP3/open sparse data with `scripts/runpod/prepare_amass_dataset.sh`.
2. For the paper-aligned standalone ADD trainer, use `scripts/runpod/train_add_physx.sh`.
3. For stock RSL-RL PPO with the same pelvis-frame sparse-pose task and dense ADD-style environment reward, use `scripts/runpod/train_rsl_physx.sh`.
4. For RSL-RL PPO plus the true online ADD adversarial discriminator, use `scripts/runpod/train_rsl_add_physx.sh`.

## G1 Flow

G1 uses sparse datasets under `data/processed/g1/`. Do not train or evaluate G1 against `data/processed/open/...`.

1. Make sure raw AIST, raw AMASS, and SMPL-H model files are available under `data/raw/...`.
2. Prepare G1 sparse data with `scripts/runpod/prepare_g1_dataset.sh`; it preprocesses AIST and AMASS directly with the G1 embodiment profile.
3. For stock G1 RSL-RL PPO, use `scripts/runpod/train_rsl_g1_physx.sh`.
4. For G1 teacher-student training, use `scripts/runpod/train_g1_teacher_student_physx.sh`.
5. Do not use `scripts/runpod/train_rsl_add_g1_physx.sh`; it intentionally exits because G1 ADD training is disabled.

On RunPod `x892ggwbr557zd`, the 2026-05-09 rebuild produced `/workspace/isaac-runpod/data/processed/g1/teleop_sparse_pose.npz` with `8,253` clips and `1,885,671` frames. That file is AMASS-derived after filtering; the regenerated AIST clips were rejected by the current G1 feasibility thresholds.

## Useful Environment Variables

- `ISAACLAB_ROOT`: path to the Isaac Lab checkout. If unset, the scripts prefer `/workspace/isaaclab`, then `/workspace/IsaacLab`.
- `ISAACLAB_REF`: branch, tag, or commit to use.
- `INSTALL_MODE`: legacy variable; active workflows should use PhysX.
- `ISAACSIM_PATH`: required for the PhysX path if you have an Isaac Sim binary install.
- `PYTHON_BIN`: optional explicit Python executable. If omitted, the RunPod scripts will use `ISAACLAB_ROOT/isaaclab.sh -p` when available, then fall back to `python` or `python3`.
- `OP3_CFG_IMPORT`: import target for your final OP3 asset config.
- `HUMANOID_TELEOP_MODE`: `synthetic` or `dataset`; preferred generic override for both OP3 and G1.
- `HUMANOID_TELEOP_DATASET_PATH`: path to the processed sparse-pose dataset NPZ; preferred generic override for both OP3 and G1.
- `G1_TELEOP_MODE`: G1-specific mode override used by G1 configs and wrappers when the generic override is unset.
- `G1_TELEOP_DATASET_PATH`: G1-specific dataset override used by G1 configs and wrappers when the generic override is unset.
- `OP3_TELEOP_MODE`: `synthetic` or `dataset`.
- `OP3_TELEOP_DATASET_PATH`: OP3-specific dataset override. G1 configs no longer read this variable.

Datasets generated before the metadata contract change should be regenerated. Current runtime loading expects the selected NPZ to declare the task embodiment and sequence FPS metadata.
