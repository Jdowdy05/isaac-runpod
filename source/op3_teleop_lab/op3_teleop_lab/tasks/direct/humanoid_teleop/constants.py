from __future__ import annotations


TRACKED_SEGMENTS = (
    "pelvis",
    "head",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)

SEGMENT_INDEX = {name: idx for idx, name in enumerate(TRACKED_SEGMENTS)}
POSE_COMPONENTS = 7
POSITION_VALID_COMPONENTS = 1
ROTATION_VALID_COMPONENTS = 1
SPARSE_POSE_DIM = (
    len(TRACKED_SEGMENTS) * POSE_COMPONENTS
    + len(TRACKED_SEGMENTS) * POSITION_VALID_COMPONENTS
    + len(TRACKED_SEGMENTS) * ROTATION_VALID_COMPONENTS
)
POSITION_CENTRIC_COMMAND_DIM = len(TRACKED_SEGMENTS) * 3 * 2 + len(TRACKED_SEGMENTS) * POSITION_VALID_COMPONENTS
