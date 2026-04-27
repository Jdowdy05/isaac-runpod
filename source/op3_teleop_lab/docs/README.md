# OP3 Teleop Lab Extension

This extension registers direct-workflow Isaac Lab tasks for sparse-pose teleoperation of humanoids. It currently
supports the Robotis OP3 and a Unitree G1 29DOF embodiment-comparison task that reuses the same sparse human command
format, the same filtered `teleop_sparse_pose.npz` dataset, and the same RSL-ADD training path.

The current task family is centered on:

- end-to-end policy learning
- unsquashed normalized joint-position actions mapped to the full controlled joint limits
- pelvis-frame sparse-pose tracking and balance
- PhysX-first execution while the OP3 Newton ground-contact issue remains unresolved
- apples-to-apples embodiment comparison with shared observations, rewards, and training code

## Registered Tasks

- `Isaac-OP3-Teleop-Direct-v0`
- `Isaac-OP3-Teleop-Newton-Direct-v0`
- `Isaac-G1-Teleop-Direct-v0`

Both tasks expose `rl_games_cfg_entry_point` and `rsl_rl_cfg_entry_point`; the RSL-RL config uses a plain Gaussian actor with learned action noise.

For RSL-RL PPO with online ADD discriminator training, use `scripts/rsl_rl/train_add.py`; this is intentionally a custom runner because stock RSL-RL PPO has no discriminator update hook.

## Important Integration Points

- `op3_teleop_lab/tasks/direct/humanoid_teleop/`: shared sparse-humanoid teleop core used by both embodiments.
- `op3_teleop_lab/tasks/direct/op3_teleop/`: OP3-specific asset resolution, profile, and task registration.
- `op3_teleop_lab/tasks/direct/g1_teleop/`: Unitree G1 29DOF profile, task registration, and runner defaults.
- `op3_teleop_lab/tasks/task_registry.py`: task metadata resolution used by generic train/play/record scripts.
- `scripts/runpod/train_rsl_add_physx.sh`: OP3 RSL-ADD RunPod wrapper.
- `scripts/runpod/train_rsl_add_g1_physx.sh`: G1 RSL-ADD RunPod wrapper for embodiment comparison.
