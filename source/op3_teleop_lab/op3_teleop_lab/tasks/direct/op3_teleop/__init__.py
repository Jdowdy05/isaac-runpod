"""Gym registration for OP3 teleoperation tasks."""

from __future__ import annotations

import gymnasium as gym


_RL_GAMES_CFG = (
    "op3_teleop_lab.tasks.direct.op3_teleop.agents:rl_games_ppo_cfg.yaml"
)
_RSL_RL_CFG = (
    "op3_teleop_lab.tasks.direct.op3_teleop.agents.rsl_rl_ppo_cfg:OP3TeleopPPORunnerCfg"
)


gym.register(
    id="Isaac-OP3-Teleop-Direct-v0",
    entry_point="op3_teleop_lab.tasks.direct.op3_teleop.env:OP3TeleopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "op3_teleop_lab.tasks.direct.op3_teleop.env_cfg:OP3TeleopEnvCfg",
        "rl_games_cfg_entry_point": _RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)

gym.register(
    id="Isaac-OP3-Teleop-Newton-Direct-v0",
    entry_point="op3_teleop_lab.tasks.direct.op3_teleop.env:OP3TeleopEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "op3_teleop_lab.tasks.direct.op3_teleop.env_cfg:OP3TeleopNewtonEnvCfg",
        "rl_games_cfg_entry_point": _RL_GAMES_CFG,
        "rsl_rl_cfg_entry_point": _RSL_RL_CFG,
    },
)
