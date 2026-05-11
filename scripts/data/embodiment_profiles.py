from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SparseEmbodimentProfile:
    name: str
    target_body_scale_m: float
    head_target: str
    aist_max_root_speed: float
    filter_max_root_speed: float
    min_pelvis_height: float
    max_pelvis_height: float
    max_torso_lean_deg: float
    min_head_height: float
    max_foot_clearance: float
    max_support_foot_clearance: float
    min_knee_height: float
    max_knee_height: float
    max_knee_to_foot: float
    max_hand_distance_from_pelvis: float
    max_feet_separation_xy: float


OP3_PROFILE = SparseEmbodimentProfile(
    name="op3",
    target_body_scale_m=0.51,
    head_target="head",
    aist_max_root_speed=0.45,
    filter_max_root_speed=0.45,
    min_pelvis_height=0.22,
    max_pelvis_height=0.34,
    max_torso_lean_deg=35.0,
    min_head_height=0.20,
    max_foot_clearance=0.055,
    max_support_foot_clearance=0.035,
    min_knee_height=0.07,
    max_knee_height=0.22,
    max_knee_to_foot=0.14,
    max_hand_distance_from_pelvis=0.36,
    max_feet_separation_xy=0.32,
)


# G1 values are based on Unitree's published proportions: about 0.6 m leg length
# (thigh + calf) and about 0.45 m arm reach, with more permissive dynamic ranges
# than OP3 because G1 is much closer to adult human embodiment.
G1_PROFILE = SparseEmbodimentProfile(
    name="g1",
    target_body_scale_m=1.02,
    head_target="upper_torso",
    aist_max_root_speed=0.90,
    filter_max_root_speed=0.90,
    min_pelvis_height=0.48,
    max_pelvis_height=0.74,
    max_torso_lean_deg=50.0,
    min_head_height=0.28,
    max_foot_clearance=0.16,
    max_support_foot_clearance=0.06,
    min_knee_height=0.10,
    max_knee_height=0.42,
    max_knee_to_foot=0.36,
    max_hand_distance_from_pelvis=0.62,
    max_feet_separation_xy=0.52,
)


EMBODIMENT_PROFILES = {
    OP3_PROFILE.name: OP3_PROFILE,
    G1_PROFILE.name: G1_PROFILE,
}


def get_embodiment_profile(name: str) -> SparseEmbodimentProfile:
    key = name.strip().lower()
    if key not in EMBODIMENT_PROFILES:
        raise ValueError(f"Unsupported embodiment {name!r}. Expected one of {tuple(sorted(EMBODIMENT_PROFILES))}.")
    return EMBODIMENT_PROFILES[key]
