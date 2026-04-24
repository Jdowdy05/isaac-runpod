# ADD Notes

This repository includes a standalone implementation of the paper:

- [Physics-Based Motion Imitation with Adversarial Differential Discriminators](https://arxiv.org/abs/2505.04961)

## What Is Implemented Here

- PPO teacher-policy training with a privileged value critic
- recurrent student distillation from the teacher policy
- a discriminator over normalized differential tracking vectors
- zero-vector positive samples for the discriminator
- discriminator replay buffer
- differential normalizer based on mean absolute value
- an RSL-RL PPO path that keeps RSL-RL's actor-critic, learned Gaussian action noise, rollout storage, clipped PPO objective, observation normalization, and adaptive KL schedule while adding online ADD discriminator rewards and discriminator updates

## Action Contract

- Teacher and student policy heads are unsquashed linear outputs, matching RSL-RL/H2O-style Gaussian actors rather than a final `tanh`.
- The environment maps action `-1`/`+1` to each joint's lower/upper position limits around the standing default, then clips the final joint-position target to those limits.
- `prev_actions` in the actor observation is the applied normalized target in `[-1, 1]`, not the unbounded raw network output.
- The standalone ADD trainer still applies a small raw action L2 penalty; the environment also penalizes raw outputs outside `[-1, 1]` so unsquashed policies do not learn to saturate invisibly.

## Observation Frame

- Deployable actor observations exclude commanded velocity and global/root-world orientation.
- Sparse pose positions are pelvis-frame coordinates. Dataset pelvis-origin world-axis deltas are rotated into the pelvis frame at load time.
- Non-pelvis sparse orientations are pelvis-relative 6D pose targets; pelvis orientation is not tracked as a global yaw command.
- Privileged critic-only features may still include simulation values such as root height and contact features.

The current OP3 integration uses the sparse teleoperation target as the reference and builds the differential vector from:

- tracked body position differences in the pelvis/root frame
- tracked body orientation differences relative to the pelvis/root frame

## RSL-RL + ADD

Use `scripts/rsl_rl/train_add.py` or `scripts/runpod/train_rsl_add_physx.sh` for the combined path. It disables the dense environment ADD reward by default and instead computes the policy reward from:

- task reward from the environment
- adversarial ADD reward `-log(1 - D(diff))`
- a small raw action L2 penalty

The discriminator is trained online every PPO update using zero-difference positives and current/replay rollout differentials as negatives.
