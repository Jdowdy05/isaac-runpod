from __future__ import annotations

from op3_teleop_lab.tasks.direct.humanoid_teleop.robot_profile import SparseHumanoidRobotProfile


def make_default_g1_profile() -> SparseHumanoidRobotProfile:
    profile = SparseHumanoidRobotProfile(
        joint_names=(
            "left_hip_pitch_joint",
            "right_hip_pitch_joint",
            "waist_yaw_joint",
            "left_hip_roll_joint",
            "right_hip_roll_joint",
            "waist_roll_joint",
            "left_hip_yaw_joint",
            "right_hip_yaw_joint",
            "waist_pitch_joint",
            "left_knee_joint",
            "right_knee_joint",
            "left_shoulder_pitch_joint",
            "right_shoulder_pitch_joint",
            "left_ankle_pitch_joint",
            "right_ankle_pitch_joint",
            "left_shoulder_roll_joint",
            "right_shoulder_roll_joint",
            "left_ankle_roll_joint",
            "right_ankle_roll_joint",
            "left_shoulder_yaw_joint",
            "right_shoulder_yaw_joint",
            "left_elbow_joint",
            "right_elbow_joint",
            "left_wrist_roll_joint",
            "right_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "right_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_wrist_yaw_joint",
        ),
        segment_to_body_name={
            "pelvis": "pelvis",
            "head": "torso_link",
            "left_hand": "left_hand_palm_link",
            "right_hand": "right_hand_palm_link",
            "left_knee": "left_knee_link",
            "right_knee": "right_knee_link",
            "left_foot": "left_ankle_roll_link",
            "right_foot": "right_ankle_roll_link",
        },
        contact_segment_to_body_name={
            "left_hand": "left_wrist_yaw_link",
            "right_hand": "right_wrist_yaw_link",
        },
        excluded_action_joint_names=(),
        termination_height=0.40,
        max_root_tilt_cos=0.55,
    )
    profile.validate()
    return profile
