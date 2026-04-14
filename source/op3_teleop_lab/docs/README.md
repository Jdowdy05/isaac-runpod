# OP3 Teleop Lab Extension

This extension registers direct-workflow Isaac Lab tasks for sparse-pose teleoperation of the Robotis OP3.

The current task family is centered on:

- end-to-end policy learning
- joint-position actions
- walking and balance
- Newton-first execution with a PhysX fallback

## Registered Tasks

- `Isaac-OP3-Teleop-Direct-v0`
- `Isaac-OP3-Teleop-Newton-Direct-v0`

## Important Integration Points

- `op3_teleop_lab/assets/op3.py`: resolves `OP3_CFG`.
- `op3_teleop_lab/tasks/direct/op3_teleop/robot_profile.py`: maps OP3 body and joint names.
- `op3_teleop_lab/tasks/direct/op3_teleop/env_cfg.py`: environment and simulator configuration.
- `op3_teleop_lab/tasks/direct/op3_teleop/env.py`: direct RL environment.

