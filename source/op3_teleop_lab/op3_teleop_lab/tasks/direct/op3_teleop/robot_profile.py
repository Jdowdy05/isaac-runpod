from __future__ import annotations

from op3_teleop_lab.tasks.direct.humanoid_teleop.robot_profile import (
    SparseHumanoidRobotProfile,
    get_action_joint_names,
)

HEAD_JOINT_NAMES = ("head_pan", "head_tilt")


def make_default_op3_profile() -> SparseHumanoidRobotProfile:
    profile = SparseHumanoidRobotProfile(
        joint_names=(
            "head_pan",
            "head_tilt",
            "l_sho_pitch",
            "l_sho_roll",
            "l_el",
            "r_sho_pitch",
            "r_sho_roll",
            "r_el",
            "l_hip_yaw",
            "l_hip_roll",
            "l_hip_pitch",
            "l_knee",
            "l_ank_pitch",
            "l_ank_roll",
            "r_hip_yaw",
            "r_hip_roll",
            "r_hip_pitch",
            "r_knee",
            "r_ank_pitch",
            "r_ank_roll",
        ),
        segment_to_body_name={
            "pelvis": "body_link",
            "head": "head_tilt_link",
            "left_hand": "l_el_link",
            "right_hand": "r_el_link",
            "left_knee": "l_knee_link",
            "right_knee": "r_knee_link",
            "left_foot": "l_ank_roll_link",
            "right_foot": "r_ank_roll_link",
        },
        excluded_action_joint_names=HEAD_JOINT_NAMES,
        termination_height=0.14,
        max_root_tilt_cos=0.55,
    )
    profile.validate()
    return profile
