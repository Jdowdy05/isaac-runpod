from __future__ import annotations

from op3_teleop_lab.tasks.direct.humanoid_teleop.env import (
    HumanoidTeleopEnv,
    quat_apply,
    quat_conjugate,
    quat_mul,
    quat_normalize,
    quaternion_to_tangent_and_normal,
)


class OP3TeleopEnv(HumanoidTeleopEnv):
    """OP3 sparse teleoperation direct-RL environment."""
