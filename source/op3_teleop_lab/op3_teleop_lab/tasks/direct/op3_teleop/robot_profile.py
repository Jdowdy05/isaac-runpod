from __future__ import annotations

from dataclasses import dataclass, field

from .constants import TRACKED_SEGMENTS

HEAD_JOINT_NAMES = ("head_pan", "head_tilt")


@dataclass
class OP3RobotProfile:
    """Robot-specific body and joint names.

    Update this file after adding the final OP3 asset so the names match your USD
    or imported articulation exactly.

    This is intentionally mutable because Isaac Lab's configclass utilities
    recurse into nested config objects during initialization.
    """

    joint_names: tuple[str, ...] = (
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
    )
    segment_to_body_name: dict[str, str] = field(
        default_factory=lambda: {
            "pelvis": "body_link",
            "head": "head_tilt_link",
            "left_hand": "l_el_link",
            "right_hand": "r_el_link",
            "left_knee": "l_knee_link",
            "right_knee": "r_knee_link",
            "left_foot": "l_ank_roll_link",
            "right_foot": "r_ank_roll_link",
        }
    )
    termination_height: float = 0.14
    max_root_tilt_cos: float = 0.55

    def validate(self) -> None:
        missing = [name for name in TRACKED_SEGMENTS if name not in self.segment_to_body_name]
        if missing:
            raise ValueError(f"Missing body-name mappings for tracked segments: {missing}")


def make_default_op3_profile() -> OP3RobotProfile:
    profile = OP3RobotProfile()
    profile.validate()
    return profile


def get_action_joint_names(profile: OP3RobotProfile) -> tuple[str, ...]:
    return tuple(name for name in profile.joint_names if name not in HEAD_JOINT_NAMES)
