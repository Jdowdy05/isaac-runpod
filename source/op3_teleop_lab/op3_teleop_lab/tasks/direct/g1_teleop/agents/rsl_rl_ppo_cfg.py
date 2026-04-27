from __future__ import annotations

from isaaclab.utils import configclass

from op3_teleop_lab.tasks.direct.op3_teleop.agents.rsl_rl_ppo_cfg import OP3TeleopPPORunnerCfg


@configclass
class G1TeleopPPORunnerCfg(OP3TeleopPPORunnerCfg):
    experiment_name = "g1_teleop_rsl_rl"
