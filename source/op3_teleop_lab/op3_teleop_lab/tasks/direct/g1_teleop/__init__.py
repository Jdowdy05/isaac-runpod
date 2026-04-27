"""Gym registration for Unitree G1 teleoperation tasks."""

from __future__ import annotations

import gymnasium as gym


_RL_GAMES_CFG = "op3_teleop_lab.tasks.direct.g1_teleop.agents:rl_games_ppo_cfg.yaml"
_RSL_RL_CFG = "op3_teleop_lab.tasks.direct.g1_teleop.agents.rsl_rl_ppo_cfg:G1TeleopPPORunnerCfg"
_ADD_CFG = "op3_teleop_lab.tasks.direct.g1_teleop.agents:add_ppo_cfg.yaml"


gym.register(
    id="Isaac-G1-Teleop-Direct-v0",
    entry_point="op3_teleop_lab.tasks.direct.g1_teleop.env:G1TeleopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "op3_teleop_lab.tasks.direct.g1_teleop.env_cfg:G1TeleopEnvCfg",
        "rl_games_cfg_entry_point": _RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
        "add_cfg_entry_point": _ADD_CFG,
        "task_slug": "g1_teleop",
    },
)
