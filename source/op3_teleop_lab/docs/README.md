# OP3 Teleop Lab Extension

This extension registers direct-workflow Isaac Lab tasks for sparse-pose teleoperation of humanoids. It currently
supports the Robotis OP3 and a Unitree G1 29DOF embodiment-comparison task. They share the sparse human command
schema and shared environment core, but they must use embodiment-specific processed datasets and training paths.

The current compatibility target is Isaac Lab `v2.3.2`, which the official Isaac Lab project documents as a
`v2.3.X` release compatible with Isaac Sim `4.5 / 5.0 / 5.1`.

The current task family is centered on:

- end-to-end policy learning
- unsquashed normalized joint-position actions mapped to the full controlled joint limits
- pelvis-frame sparse-pose tracking, sparse orientation targets, local root-velocity commands, and balance
- PhysX-only execution on Isaac Sim
- embodiment comparison with shared observations and rewards but separate processed sparse datasets

## Registered Tasks

- `Isaac-OP3-Teleop-Direct-v0`
- `Isaac-G1-Teleop-Direct-v0`

Both tasks expose `rl_games_cfg_entry_point` and `rsl_rl_cfg_entry_point`; the RSL-RL config uses a plain Gaussian actor with learned action noise.

For OP3 RSL-RL PPO with online ADD discriminator training, use `scripts/rsl_rl/train_add.py`; this is intentionally a custom runner because stock RSL-RL PPO has no discriminator update hook. G1 ADD training is disabled for now; use stock G1 PPO or the G1 teacher-student path.

## Important Integration Points

- `op3_teleop_lab/tasks/direct/humanoid_teleop/`: shared sparse-humanoid teleop core used by both embodiments.
- `op3_teleop_lab/tasks/direct/op3_teleop/`: OP3-specific asset resolution, profile, and task registration.
- `op3_teleop_lab/tasks/direct/g1_teleop/`: Unitree G1 29DOF profile, task registration, and runner defaults.
- `op3_teleop_lab/tasks/task_registry.py`: task metadata resolution used by generic train/play/record scripts.
- `scripts/runpod/train_rsl_add_physx.sh`: OP3 RSL-ADD RunPod wrapper.
- `scripts/runpod/train_rsl_g1_physx.sh`: stock G1 RSL-RL PPO RunPod wrapper without ADD.
- `scripts/runpod/train_g1_teacher_student_physx.sh`: G1 teacher-student RunPod wrapper.
- Sparse datasets must include `embodiment`, per-sequence FPS metadata, and root velocity commands. The command loader rejects missing embodiment metadata and plays sequences by elapsed time rather than assuming all clips are 50 Hz.
