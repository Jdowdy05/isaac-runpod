from __future__ import annotations

from dataclasses import dataclass, field
import re

from .constants import TRACKED_SEGMENTS

CONTACT_SEGMENT_NAMES = (
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)


@dataclass
class SparseHumanoidRobotProfile:
    """Robot-specific sparse-body and joint naming for humanoid teleop tasks."""

    joint_names: tuple[str, ...] = ()
    segment_to_body_name: dict[str, str] = field(default_factory=dict)
    contact_segment_to_body_name: dict[str, str] = field(default_factory=dict)
    excluded_action_joint_names: tuple[str, ...] = ()
    contact_segment_names: tuple[str, ...] = CONTACT_SEGMENT_NAMES
    termination_height: float = 0.14
    max_root_tilt_cos: float = 0.55

    def validate(self) -> None:
        missing = [name for name in TRACKED_SEGMENTS if name not in self.segment_to_body_name]
        if missing:
            raise ValueError(f"Missing body-name mappings for tracked segments: {missing}")
        missing_contacts = [
            name
            for name in self.contact_segment_names
            if name not in self.segment_to_body_name and name not in self.contact_segment_to_body_name
        ]
        if missing_contacts:
            raise ValueError(f"Missing body-name mappings for contact segments: {missing_contacts}")

    def contact_body_name_for(self, segment_name: str) -> str:
        return self.contact_segment_to_body_name.get(segment_name, self.segment_to_body_name[segment_name])

    def contact_body_names(self) -> tuple[str, ...]:
        return tuple(self.contact_body_name_for(name) for name in self.contact_segment_names)

    def contact_sensor_body_regex(self) -> str:
        return "|".join(re.escape(name) for name in self.contact_body_names())


def get_action_joint_names(profile: SparseHumanoidRobotProfile) -> tuple[str, ...]:
    return tuple(name for name in profile.joint_names if name not in profile.excluded_action_joint_names)
