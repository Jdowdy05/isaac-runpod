# OP3 Teleop Lab Extension

This extension registers direct-workflow Isaac Lab tasks for sparse-pose teleoperation of the Robotis OP3.

The current task family is centered on:

- end-to-end policy learning
- unsquashed normalized joint-position actions mapped to the full controlled joint limits
- pelvis-frame sparse-pose tracking and balance
- PhysX-first execution while the OP3 Newton ground-contact issue remains unresolved

## Registered Tasks

- `Isaac-OP3-Teleop-Direct-v0`
- `Isaac-OP3-Teleop-Newton-Direct-v0`

Both tasks expose `rl_games_cfg_entry_point` and `rsl_rl_cfg_entry_point`; the RSL-RL config uses a plain Gaussian actor with learned action noise.

For RSL-RL PPO with online ADD discriminator training, use `scripts/rsl_rl/train_add.py`; this is intentionally a custom runner because stock RSL-RL PPO has no discriminator update hook.

## Important Integration Points

- `op3_teleop_lab/assets/op3.py`: resolves `OP3_CFG`.
- `op3_teleop_lab/tasks/direct/op3_teleop/robot_profile.py`: maps OP3 body and joint names.
- `op3_teleop_lab/tasks/direct/op3_teleop/env_cfg.py`: environment and simulator configuration.
- `op3_teleop_lab/tasks/direct/op3_teleop/env.py`: direct RL environment.
