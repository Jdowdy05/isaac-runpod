#!/usr/bin/env python3

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch


DEFAULT_SUBSETS = (
    "ACCAD",
    "BMLmovi",
    "BMLrub",
    "CMU",
    "EKUT",
    "EyesJapanDataset",
    "HDM05",
    "HumanEva",
    "KIT",
    "TotalCapture",
    "Transitions",
    "DanceDB",
)

SUBSET_ALIASES = {
    "ACCAD": ("ACCAD",),
    "BMLhandball": ("BMLhandball",),
    "BMLmovi": ("BMLmovi",),
    "BMLrub": ("BMLrub", "BioMotionLab_NTroje"),
    "CMU": ("CMU",),
    "CNRS": ("CNRS",),
    "DanceDB": ("DanceDB",),
    "DFaust": ("DFaust", "DFaust_67"),
    "EKUT": ("EKUT",),
    "EyesJapanDataset": ("EyesJapanDataset", "Eyes_Japan_Dataset"),
    "GRAB": ("GRAB",),
    "HDM05": ("HDM05", "MPI_HDM05"),
    "HUMAN4D": ("HUMAN4D",),
    "HumanEva": ("HumanEva",),
    "KIT": ("KIT",),
    "MoSh": ("MoSh", "MPI_mosh"),
    "PosePrior": ("PosePrior",),
    "SFU": ("SFU",),
    "SOMA": ("SOMA",),
    "SSM": ("SSM", "SSM_synced"),
    "TCDHands": ("TCDHands", "TCD_handMocap"),
    "TotalCapture": ("TotalCapture",),
    "Transitions": ("Transitions", "Transitions_mocap"),
    "WEIZMANN": ("WEIZMANN",),
}

SMPLH_BODY_JOINTS = {
    "pelvis": 0,
    "left_hip": 1,
    "right_hip": 2,
    "left_knee": 4,
    "right_knee": 5,
    "left_ankle": 7,
    "right_ankle": 8,
    "left_foot": 10,
    "right_foot": 11,
    "neck": 12,
    "head": 15,
    "left_shoulder": 16,
    "right_shoulder": 17,
    "left_elbow": 18,
    "right_elbow": 19,
    "left_wrist": 20,
    "right_wrist": 21,
}

SEGMENTS = (
    "pelvis",
    "head",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)
SEGMENT_INDEX = {name: idx for idx, name in enumerate(SEGMENTS)}
OP3_TARGET_BODY_SCALE_M = 0.51


@dataclass(frozen=True)
class MotionFilterConfig:
    clip_frames: int
    clip_stride_frames: int
    min_contact_ratio: float
    max_double_air_ratio: float
    max_root_speed: float
    max_root_yaw_rate: float
    max_root_jerk: float
    min_pelvis_height: float
    max_pelvis_height: float
    max_torso_lean_deg: float
    max_foot_clearance: float
    max_hand_reach: float
    max_feet_separation_xy: float
    contact_height_threshold: float
    contact_speed_threshold: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AMASS SMPL+H motion files into sparse teleoperation commands.")
    parser.add_argument("--amass-root", type=Path, required=True)
    parser.add_argument("--smpl-model-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--target-fps", type=float, default=50.0)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--subsets", nargs="*", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument(
        "--disable-feasibility-filter",
        action="store_true",
        help="Keep all AMASS sequences after sparse conversion without OP3-specific clip filtering.",
    )
    parser.add_argument("--filter-clip-seconds", type=float, default=4.0)
    parser.add_argument("--filter-stride-seconds", type=float, default=4.0)
    parser.add_argument("--min-contact-ratio", type=float, default=0.75)
    parser.add_argument("--max-double-air-ratio", type=float, default=0.08)
    parser.add_argument("--max-root-speed", type=float, default=0.45)
    parser.add_argument("--max-root-yaw-rate", type=float, default=2.5)
    parser.add_argument("--max-root-jerk", type=float, default=35.0)
    parser.add_argument("--min-pelvis-height", type=float, default=0.17)
    parser.add_argument("--max-pelvis-height", type=float, default=0.36)
    parser.add_argument("--max-torso-lean-deg", type=float, default=45.0)
    parser.add_argument("--max-foot-clearance", type=float, default=0.16)
    parser.add_argument("--max-hand-reach", type=float, default=0.24)
    parser.add_argument("--max-feet-separation-xy", type=float, default=0.32)
    parser.add_argument("--contact-height-threshold", type=float, default=0.025)
    parser.add_argument("--contact-speed-threshold", type=float, default=0.18)
    return parser.parse_args()


def decode_gender(value) -> str:
    if isinstance(value, np.ndarray) and value.shape == ():
        value = value.item()
    if isinstance(value, bytes):
        value = value.decode("utf-8")
    value = str(value).strip().lower()
    if value in {"male", "female", "neutral"}:
        return value
    return "neutral"


def finite_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def forward_fill(points: np.ndarray) -> np.ndarray:
    output = points.copy()
    valid = np.isfinite(output).all(axis=-1)
    if not np.any(valid):
        return np.zeros_like(output)
    first_valid = int(np.argmax(valid))
    output[:first_valid] = output[first_valid]
    for idx in range(first_valid + 1, len(output)):
        if not valid[idx]:
            output[idx] = output[idx - 1]
    return output


def estimate_body_scale(
    pelvis: np.ndarray,
    head: np.ndarray,
    left_foot: np.ndarray,
    right_foot: np.ndarray,
) -> float:
    valid = finite_mask(pelvis) & finite_mask(head) & finite_mask(left_foot) & finite_mask(right_foot)
    if not np.any(valid):
        return 1.0

    torso = np.linalg.norm(head[valid] - pelvis[valid], axis=-1)
    left_leg = np.linalg.norm(left_foot[valid] - pelvis[valid], axis=-1)
    right_leg = np.linalg.norm(right_foot[valid] - pelvis[valid], axis=-1)
    body_scale = torso + 0.5 * (left_leg + right_leg)
    scale = float(np.quantile(body_scale.astype(np.float32), 0.9))
    return max(scale, 1.0e-3)


def compute_op3_scale_from_joints(joints: np.ndarray) -> np.float32:
    pelvis = joints[:, SMPLH_BODY_JOINTS["pelvis"]]
    head = joints[:, SMPLH_BODY_JOINTS["head"]]
    left_ankle = joints[:, SMPLH_BODY_JOINTS["left_ankle"]]
    right_ankle = joints[:, SMPLH_BODY_JOINTS["right_ankle"]]
    body_scale = estimate_body_scale(pelvis, head, left_ankle, right_ankle)
    return np.float32(OP3_TARGET_BODY_SCALE_M / body_scale)


def normalize_vectors(vectors: np.ndarray, eps: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    valid = norms[..., 0] > eps
    normalized = np.zeros_like(vectors, dtype=np.float32)
    normalized[valid] = vectors[valid] / norms[valid]
    return normalized, valid


def make_frame_from_forward_up(forward_hint: np.ndarray, up_hint: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x_axis, x_valid = normalize_vectors(forward_hint)
    y_axis, y_valid = normalize_vectors(np.cross(up_hint, x_axis))
    z_axis, z_valid = normalize_vectors(np.cross(x_axis, y_axis))
    mats = np.stack((x_axis, y_axis, z_axis), axis=-1).astype(np.float32)
    valid = x_valid & y_valid & z_valid
    mats[~valid] = np.eye(3, dtype=np.float32)
    return mats, valid


def rotation_matrices_to_quats_xyzw(mats: np.ndarray) -> np.ndarray:
    quats = np.zeros((mats.shape[0], 4), dtype=np.float32)
    trace = mats[:, 0, 0] + mats[:, 1, 1] + mats[:, 2, 2]

    positive = trace > 0.0
    if np.any(positive):
        s = np.sqrt(trace[positive] + 1.0) * 2.0
        quats[positive, 0] = (mats[positive, 2, 1] - mats[positive, 1, 2]) / s
        quats[positive, 1] = (mats[positive, 0, 2] - mats[positive, 2, 0]) / s
        quats[positive, 2] = (mats[positive, 1, 0] - mats[positive, 0, 1]) / s
        quats[positive, 3] = 0.25 * s

    cond_x = (~positive) & (mats[:, 0, 0] > mats[:, 1, 1]) & (mats[:, 0, 0] > mats[:, 2, 2])
    if np.any(cond_x):
        s = np.sqrt(1.0 + mats[cond_x, 0, 0] - mats[cond_x, 1, 1] - mats[cond_x, 2, 2]) * 2.0
        quats[cond_x, 0] = 0.25 * s
        quats[cond_x, 1] = (mats[cond_x, 0, 1] + mats[cond_x, 1, 0]) / s
        quats[cond_x, 2] = (mats[cond_x, 0, 2] + mats[cond_x, 2, 0]) / s
        quats[cond_x, 3] = (mats[cond_x, 2, 1] - mats[cond_x, 1, 2]) / s

    cond_y = (~positive) & (~cond_x) & (mats[:, 1, 1] > mats[:, 2, 2])
    if np.any(cond_y):
        s = np.sqrt(1.0 + mats[cond_y, 1, 1] - mats[cond_y, 0, 0] - mats[cond_y, 2, 2]) * 2.0
        quats[cond_y, 0] = (mats[cond_y, 0, 1] + mats[cond_y, 1, 0]) / s
        quats[cond_y, 1] = 0.25 * s
        quats[cond_y, 2] = (mats[cond_y, 1, 2] + mats[cond_y, 2, 1]) / s
        quats[cond_y, 3] = (mats[cond_y, 0, 2] - mats[cond_y, 2, 0]) / s

    cond_z = (~positive) & (~cond_x) & (~cond_y)
    if np.any(cond_z):
        s = np.sqrt(1.0 + mats[cond_z, 2, 2] - mats[cond_z, 0, 0] - mats[cond_z, 1, 1]) * 2.0
        quats[cond_z, 0] = (mats[cond_z, 0, 2] + mats[cond_z, 2, 0]) / s
        quats[cond_z, 1] = (mats[cond_z, 1, 2] + mats[cond_z, 2, 1]) / s
        quats[cond_z, 2] = 0.25 * s
        quats[cond_z, 3] = (mats[cond_z, 1, 0] - mats[cond_z, 0, 1]) / s

    norms = np.linalg.norm(quats, axis=-1, keepdims=True)
    quats /= np.clip(norms, 1.0e-6, None)
    return quats


def build_filter_config(args: argparse.Namespace, effective_fps: float) -> MotionFilterConfig:
    clip_frames = max(int(round(args.filter_clip_seconds * effective_fps)), args.min_frames)
    clip_stride_frames = max(int(round(args.filter_stride_seconds * effective_fps)), 1)
    return MotionFilterConfig(
        clip_frames=clip_frames,
        clip_stride_frames=clip_stride_frames,
        min_contact_ratio=float(args.min_contact_ratio),
        max_double_air_ratio=float(args.max_double_air_ratio),
        max_root_speed=float(args.max_root_speed),
        max_root_yaw_rate=float(args.max_root_yaw_rate),
        max_root_jerk=float(args.max_root_jerk),
        min_pelvis_height=float(args.min_pelvis_height),
        max_pelvis_height=float(args.max_pelvis_height),
        max_torso_lean_deg=float(args.max_torso_lean_deg),
        max_foot_clearance=float(args.max_foot_clearance),
        max_hand_reach=float(args.max_hand_reach),
        max_feet_separation_xy=float(args.max_feet_separation_xy),
        contact_height_threshold=float(args.contact_height_threshold),
        contact_speed_threshold=float(args.contact_speed_threshold),
    )


def iter_clip_ranges(num_frames: int, clip_frames: int, stride_frames: int, min_frames: int) -> list[tuple[int, int]]:
    if num_frames < min_frames:
        return []
    if num_frames <= clip_frames:
        return [(0, num_frames)]

    clip_ranges: list[tuple[int, int]] = []
    start = 0
    while start + clip_frames <= num_frames:
        clip_ranges.append((start, start + clip_frames))
        start += stride_frames
    if not clip_ranges:
        return [(0, num_frames)]
    return clip_ranges


def contact_mask_from_foot(
    foot_positions: np.ndarray,
    effective_fps: float,
    ground_height: float,
    height_threshold: float,
    speed_threshold: float,
) -> np.ndarray:
    foot_speed = np.linalg.norm(
        np.diff(foot_positions, axis=0, prepend=foot_positions[:1]) * effective_fps,
        axis=-1,
    )
    foot_height = foot_positions[:, 2] - ground_height
    return (foot_height <= height_threshold) & (foot_speed <= speed_threshold)


def filter_motion_clips(
    joints: np.ndarray,
    effective_fps: float,
    op3_scale: np.float32,
    cfg: MotionFilterConfig,
) -> tuple[list[tuple[int, int]], Counter[str]]:
    scaled_joints = joints.astype(np.float32) * float(op3_scale)

    pelvis = scaled_joints[:, SMPLH_BODY_JOINTS["pelvis"]]
    left_hip = scaled_joints[:, SMPLH_BODY_JOINTS["left_hip"]]
    right_hip = scaled_joints[:, SMPLH_BODY_JOINTS["right_hip"]]
    left_shoulder = scaled_joints[:, SMPLH_BODY_JOINTS["left_shoulder"]]
    right_shoulder = scaled_joints[:, SMPLH_BODY_JOINTS["right_shoulder"]]
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)
    left_wrist = scaled_joints[:, SMPLH_BODY_JOINTS["left_wrist"]]
    right_wrist = scaled_joints[:, SMPLH_BODY_JOINTS["right_wrist"]]
    left_foot = scaled_joints[:, SMPLH_BODY_JOINTS["left_foot"]]
    right_foot = scaled_joints[:, SMPLH_BODY_JOINTS["right_foot"]]

    ground_height = float(np.quantile(np.concatenate((left_foot[:, 2], right_foot[:, 2])), 0.05))
    left_contact = contact_mask_from_foot(
        left_foot,
        effective_fps=effective_fps,
        ground_height=ground_height,
        height_threshold=cfg.contact_height_threshold,
        speed_threshold=cfg.contact_speed_threshold,
    )
    right_contact = contact_mask_from_foot(
        right_foot,
        effective_fps=effective_fps,
        ground_height=ground_height,
        height_threshold=cfg.contact_height_threshold,
        speed_threshold=cfg.contact_speed_threshold,
    )
    contact_any = left_contact | right_contact
    double_air = ~contact_any

    root_vel = np.diff(pelvis[:, :2], axis=0, prepend=pelvis[:1, :2]) * effective_fps
    root_speed = np.linalg.norm(root_vel, axis=-1)
    if len(root_vel) >= 2:
        root_acc = np.diff(root_vel, axis=0, prepend=root_vel[:1]) * effective_fps
    else:
        root_acc = np.zeros_like(root_vel)
    if len(root_acc) >= 2:
        root_jerk = np.linalg.norm(np.diff(root_acc, axis=0, prepend=root_acc[:1]) * effective_fps, axis=-1)
    else:
        root_jerk = np.zeros(len(pelvis), dtype=np.float32)

    pelvis_lateral = (left_hip - right_hip) + (left_shoulder - right_shoulder)
    pelvis_up_hint = shoulder_center - pelvis
    pelvis_forward_hint = np.cross(pelvis_lateral, pelvis_up_hint)
    pelvis_world, _ = make_frame_from_forward_up(pelvis_forward_hint, pelvis_up_hint)
    pelvis_forward = pelvis_world[:, :, 0]
    pelvis_heading = np.unwrap(np.arctan2(pelvis_forward[:, 1], pelvis_forward[:, 0]))
    root_yaw_rate = np.abs(np.diff(pelvis_heading, prepend=pelvis_heading[:1]) * effective_fps)

    torso_up, torso_valid = normalize_vectors(shoulder_center - pelvis)
    torso_cos = np.clip(torso_up[:, 2], -1.0, 1.0)
    torso_lean_deg = np.degrees(np.arccos(torso_cos)).astype(np.float32)
    torso_lean_deg[~torso_valid] = 0.0

    pelvis_height = pelvis[:, 2] - ground_height
    foot_clearance = np.maximum(left_foot[:, 2], right_foot[:, 2]) - ground_height
    hand_reach = np.maximum(
        np.linalg.norm(left_wrist - left_shoulder, axis=-1),
        np.linalg.norm(right_wrist - right_shoulder, axis=-1),
    )
    feet_separation_xy = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=-1)

    kept_clips: list[tuple[int, int]] = []
    rejection_counts: Counter[str] = Counter()
    clip_ranges = iter_clip_ranges(
        num_frames=len(joints),
        clip_frames=cfg.clip_frames,
        stride_frames=cfg.clip_stride_frames,
        min_frames=min(cfg.clip_frames, len(joints)),
    )
    for start, end in clip_ranges:
        clip = slice(start, end)
        reasons: list[str] = []
        if float(np.max(root_speed[clip])) > cfg.max_root_speed:
            reasons.append("root_speed")
        if float(np.max(root_yaw_rate[clip])) > cfg.max_root_yaw_rate:
            reasons.append("root_yaw_rate")
        if float(np.max(root_jerk[clip])) > cfg.max_root_jerk:
            reasons.append("root_jerk")
        if float(np.min(pelvis_height[clip])) < cfg.min_pelvis_height:
            reasons.append("pelvis_too_low")
        if float(np.max(pelvis_height[clip])) > cfg.max_pelvis_height:
            reasons.append("pelvis_too_high")
        if float(np.max(torso_lean_deg[clip])) > cfg.max_torso_lean_deg:
            reasons.append("torso_lean")
        if float(np.max(foot_clearance[clip])) > cfg.max_foot_clearance:
            reasons.append("foot_clearance")
        if float(np.max(hand_reach[clip])) > cfg.max_hand_reach:
            reasons.append("hand_reach")
        if float(np.max(feet_separation_xy[clip])) > cfg.max_feet_separation_xy:
            reasons.append("feet_separation")
        if float(np.mean(contact_any[clip])) < cfg.min_contact_ratio:
            reasons.append("contact_ratio")
        if float(np.mean(double_air[clip])) > cfg.max_double_air_ratio:
            reasons.append("double_air")

        if reasons:
            rejection_counts.update(reasons)
            continue
        kept_clips.append((start, end))

    return kept_clips, rejection_counts


def build_sparse_sequence_from_joints(
    joints: np.ndarray,
    effective_fps: float,
    op3_scale: np.float32 | None = None,
) -> tuple[np.ndarray, ...]:
    num_frames = joints.shape[0]
    positions = np.zeros((num_frames, len(SEGMENTS), 3), dtype=np.float32)
    orientations = np.zeros((num_frames, len(SEGMENTS), 4), dtype=np.float32)
    orientations[..., 3] = 1.0
    position_valid = np.ones((num_frames, len(SEGMENTS)), dtype=bool)
    rotation_valid = np.zeros((num_frames, len(SEGMENTS)), dtype=bool)

    pelvis = joints[:, SMPLH_BODY_JOINTS["pelvis"]]
    head = joints[:, SMPLH_BODY_JOINTS["head"]]
    neck = joints[:, SMPLH_BODY_JOINTS["neck"]]
    left_shoulder = joints[:, SMPLH_BODY_JOINTS["left_shoulder"]]
    right_shoulder = joints[:, SMPLH_BODY_JOINTS["right_shoulder"]]
    left_elbow = joints[:, SMPLH_BODY_JOINTS["left_elbow"]]
    right_elbow = joints[:, SMPLH_BODY_JOINTS["right_elbow"]]
    left_wrist = joints[:, SMPLH_BODY_JOINTS["left_wrist"]]
    right_wrist = joints[:, SMPLH_BODY_JOINTS["right_wrist"]]
    left_hip = joints[:, SMPLH_BODY_JOINTS["left_hip"]]
    right_hip = joints[:, SMPLH_BODY_JOINTS["right_hip"]]
    left_knee = joints[:, SMPLH_BODY_JOINTS["left_knee"]]
    right_knee = joints[:, SMPLH_BODY_JOINTS["right_knee"]]
    left_ankle = joints[:, SMPLH_BODY_JOINTS["left_ankle"]]
    right_ankle = joints[:, SMPLH_BODY_JOINTS["right_ankle"]]
    left_foot = joints[:, SMPLH_BODY_JOINTS["left_foot"]]
    right_foot = joints[:, SMPLH_BODY_JOINTS["right_foot"]]
    if op3_scale is None:
        op3_scale = compute_op3_scale_from_joints(joints)

    raw_targets = {
        "pelvis": pelvis,
        "head": head,
        "left_hand": left_wrist,
        "right_hand": right_wrist,
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_foot": left_ankle,
        "right_foot": right_ankle,
    }

    pelvis_filled = forward_fill(pelvis)
    for segment_name, points in raw_targets.items():
        seg_idx = SEGMENT_INDEX[segment_name]
        positions[:, seg_idx] = points - pelvis_filled
        position_valid[:, seg_idx] = finite_mask(points)

    positions[:, SEGMENT_INDEX["pelvis"]] = 0.0
    positions *= op3_scale
    target_lin_vel_xy = np.diff(pelvis_filled[:, :2], axis=0, prepend=pelvis_filled[:1, :2]) * effective_fps
    target_lin_vel_xy = target_lin_vel_xy.astype(np.float32) * op3_scale

    pelvis_lateral = (left_hip - right_hip) + (left_shoulder - right_shoulder)
    pelvis_up_hint = 0.5 * (left_shoulder + right_shoulder) - pelvis
    pelvis_forward_hint = np.cross(pelvis_lateral, pelvis_up_hint)
    pelvis_world, pelvis_rot_valid = make_frame_from_forward_up(pelvis_forward_hint, pelvis_up_hint)
    pelvis_world_inv = np.swapaxes(pelvis_world, -1, -2)
    pelvis_forward = pelvis_world[:, :, 0]
    pelvis_up = pelvis_world[:, :, 2]

    orientations[:, SEGMENT_INDEX["pelvis"]] = rotation_matrices_to_quats_xyzw(pelvis_world)
    rotation_valid[:, SEGMENT_INDEX["pelvis"]] = pelvis_rot_valid

    def set_relative_orientation(segment_name: str, world_mat: np.ndarray, valid_mask: np.ndarray) -> None:
        seg_idx = SEGMENT_INDEX[segment_name]
        relative_mat = np.einsum("tij,tjk->tik", pelvis_world_inv, world_mat).astype(np.float32)
        orientations[:, seg_idx] = rotation_matrices_to_quats_xyzw(relative_mat)
        rotation_valid[:, seg_idx] = valid_mask & pelvis_rot_valid

    head_world, head_rot_valid = make_frame_from_forward_up(
        np.cross(left_shoulder - right_shoulder, head - neck),
        head - neck,
    )
    set_relative_orientation("head", head_world, head_rot_valid)

    left_hand_world, left_hand_rot_valid = make_frame_from_forward_up(left_wrist - left_elbow, pelvis_up)
    set_relative_orientation("left_hand", left_hand_world, left_hand_rot_valid)

    right_hand_world, right_hand_rot_valid = make_frame_from_forward_up(right_wrist - right_elbow, pelvis_up)
    set_relative_orientation("right_hand", right_hand_world, right_hand_rot_valid)

    left_knee_world, left_knee_rot_valid = make_frame_from_forward_up(left_ankle - left_knee, pelvis_up)
    set_relative_orientation("left_knee", left_knee_world, left_knee_rot_valid)

    right_knee_world, right_knee_rot_valid = make_frame_from_forward_up(right_ankle - right_knee, pelvis_up)
    set_relative_orientation("right_knee", right_knee_world, right_knee_rot_valid)

    left_foot_world, left_foot_rot_valid = make_frame_from_forward_up(left_foot - left_ankle, left_knee - left_ankle)
    set_relative_orientation("left_foot", left_foot_world, left_foot_rot_valid)

    right_foot_world, right_foot_rot_valid = make_frame_from_forward_up(right_foot - right_ankle, right_knee - right_ankle)
    set_relative_orientation("right_foot", right_foot_world, right_foot_rot_valid)

    pelvis_rot_missing = ~rotation_valid[:, SEGMENT_INDEX["pelvis"]]
    if np.any(pelvis_rot_missing):
        fallback_world, fallback_valid = make_frame_from_forward_up(pelvis_forward, pelvis_up)
        orientations[pelvis_rot_missing, SEGMENT_INDEX["pelvis"]] = rotation_matrices_to_quats_xyzw(
            fallback_world[pelvis_rot_missing]
        )
        rotation_valid[pelvis_rot_missing, SEGMENT_INDEX["pelvis"]] = fallback_valid[pelvis_rot_missing]

    return positions, orientations, position_valid, rotation_valid, target_lin_vel_xy


def load_motion_arrays(data: np.lib.npyio.NpzFile, target_fps: float) -> tuple[np.ndarray, ...]:
    if "mocap_framerate" in data:
        fps = float(data["mocap_framerate"])
    elif "mocap_frame_rate" in data:
        fps = float(data["mocap_frame_rate"])
    else:
        fps = 60.0
    stride = max(1, int(round(fps / target_fps)))

    if "poses" in data:
        poses = np.asarray(data["poses"], dtype=np.float32)[::stride]
        trans = np.asarray(data["trans"], dtype=np.float32)[::stride]
        if poses.shape[-1] < 66:
            raise ValueError(f"Unsupported AMASS pose width: {poses.shape[-1]}")
        root_orient = poses[:, :3]
        body_pose = poses[:, 3:66]
        left_hand_pose = np.zeros((len(poses), 45), dtype=np.float32)
        right_hand_pose = np.zeros((len(poses), 45), dtype=np.float32)
        if poses.shape[-1] >= 156:
            left_hand_pose = poses[:, 66:111]
            right_hand_pose = poses[:, 111:156]
    else:
        root_orient = np.asarray(data["root_orient"], dtype=np.float32)[::stride]
        body_pose = np.asarray(data["pose_body"], dtype=np.float32)[::stride]
        trans = np.asarray(data["trans"], dtype=np.float32)[::stride]
        if "pose_hand" in data:
            pose_hand = np.asarray(data["pose_hand"], dtype=np.float32)[::stride]
            left_hand_pose = pose_hand[:, :45]
            right_hand_pose = pose_hand[:, 45:90]
        else:
            left_hand_pose = np.zeros((len(root_orient), 45), dtype=np.float32)
            right_hand_pose = np.zeros((len(root_orient), 45), dtype=np.float32)

    effective_fps = fps / float(stride)
    return root_orient, body_pose, left_hand_pose, right_hand_pose, trans, effective_fps


def resolve_model_ext(model_root: Path) -> str:
    if any(model_root.glob("SMPLH_*.pkl")) or any((model_root / "smplh").glob("SMPLH_*.pkl")):
        return "pkl"
    if any(model_root.glob("SMPLH_*.npz")) or any((model_root / "smplh").glob("SMPLH_*.npz")):
        return "npz"
    raise FileNotFoundError(
        "Could not find SMPL-H body model files. "
        f"Expected files like SMPLH_MALE/FEMALE/NEUTRAL under {model_root} or {model_root / 'smplh'}. "
        "The AMASS 'SMPL+H G' motion datasets are separate and do not satisfy this requirement."
    )


def build_model_cache(model_root: Path, model_ext: str, device: torch.device):
    try:
        import smplx  # type: ignore
    except ImportError as exc:
        raise ImportError(
            "prepare_amass_sparse.py requires the 'smplx' package. Re-run scripts/runpod/bootstrap.sh after pulling."
        ) from exc

    cache: dict[str, object] = {}
    model_base = model_root
    if any(model_root.glob("SMPLH_*.pkl")) or any(model_root.glob("SMPLH_*.npz")):
        if model_root.name.lower() == "smplh":
            model_base = model_root.parent
    elif not (model_root / "smplh").is_dir():
        raise FileNotFoundError(
            "Could not locate a compatible SMPL-H model directory layout for smplx. "
            f"Checked {model_root} and {model_root / 'smplh'}."
        )

    available_genders = {
        path.stem.removeprefix("SMPLH_").lower()
        for path in model_root.glob(f"SMPLH_*.{model_ext}")
    }
    available_genders.update(
        path.stem.removeprefix("SMPLH_").lower()
        for path in (model_root / "smplh").glob(f"SMPLH_*.{model_ext}")
    )
    if not available_genders:
        raise FileNotFoundError(
            f"Could not find any SMPL-H model files with extension .{model_ext} under {model_root}."
        )

    fallback_gender = (
        "neutral"
        if "neutral" in available_genders
        else "male"
        if "male" in available_genders
        else "female"
    )

    def get_model(gender: str):
        resolved_gender = gender if gender in available_genders else fallback_gender
        if resolved_gender not in cache:
            cache[resolved_gender] = smplx.create(
                str(model_base),
                model_type="smplh",
                gender=resolved_gender,
                ext=model_ext,
                use_pca=False,
                flat_hand_mean=True,
            ).to(device)
        return cache[resolved_gender]

    return get_model


def iter_motion_files(amass_root: Path, subsets: list[str]) -> list[Path]:
    subset_paths: list[Path] = []
    for subset in subsets:
        aliases = SUBSET_ALIASES.get(subset, (subset,))
        for alias in aliases:
            candidate = amass_root / alias
            if candidate.exists():
                subset_paths.append(candidate)
                break
    motion_paths: list[Path] = []
    for subset_path in subset_paths:
        motion_paths.extend(sorted(path for path in subset_path.rglob("*.npz") if path.name != "shape.npz"))
    return motion_paths


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_ext = resolve_model_ext(args.smpl_model_root)
    get_model = build_model_cache(args.smpl_model_root, model_ext, device)

    position_blocks: list[np.ndarray] = []
    orientation_blocks: list[np.ndarray] = []
    position_valid_blocks: list[np.ndarray] = []
    rotation_valid_blocks: list[np.ndarray] = []
    velocity_blocks: list[np.ndarray] = []
    sequence_starts: list[int] = []
    sequence_lengths: list[int] = []
    sources: list[str] = []
    total_frames = 0
    rejected_clips = 0
    kept_clips = 0
    rejection_reasons: Counter[str] = Counter()

    motion_paths = iter_motion_files(args.amass_root, args.subsets)
    if args.limit is not None:
        motion_paths = motion_paths[: args.limit]
    if not motion_paths:
        raise RuntimeError(f"No AMASS motion files found under {args.amass_root} for subsets {args.subsets}")

    for motion_path in motion_paths:
        data = np.load(motion_path, allow_pickle=True)
        gender = decode_gender(data["gender"] if "gender" in data else "neutral")
        betas = np.asarray(data["betas"], dtype=np.float32)
        if betas.shape[0] < 16:
            betas = np.pad(betas, (0, 16 - betas.shape[0]))
        betas = betas[:16]
        root_orient, body_pose, left_hand_pose, right_hand_pose, trans, effective_fps = load_motion_arrays(
            data,
            target_fps=args.target_fps,
        )

        if len(root_orient) < args.min_frames:
            continue

        model = get_model(gender)
        model_num_betas = int(getattr(model, "num_betas", model.shapedirs.shape[-1]))
        joints_batches: list[np.ndarray] = []
        clip_betas = betas[:model_num_betas]
        if clip_betas.shape[0] < model_num_betas:
            clip_betas = np.pad(clip_betas, (0, model_num_betas - clip_betas.shape[0]))
        betas_batch = np.broadcast_to(clip_betas, (min(args.chunk_size, len(root_orient)), model_num_betas)).copy()

        with torch.no_grad():
            for start in range(0, len(root_orient), args.chunk_size):
                end = min(start + args.chunk_size, len(root_orient))
                batch_size = end - start
                joints = model(
                    global_orient=torch.as_tensor(root_orient[start:end], dtype=torch.float32, device=device),
                    body_pose=torch.as_tensor(body_pose[start:end], dtype=torch.float32, device=device),
                    left_hand_pose=torch.as_tensor(left_hand_pose[start:end], dtype=torch.float32, device=device),
                    right_hand_pose=torch.as_tensor(right_hand_pose[start:end], dtype=torch.float32, device=device),
                    transl=torch.as_tensor(trans[start:end], dtype=torch.float32, device=device),
                    betas=torch.as_tensor(betas_batch[:batch_size], dtype=torch.float32, device=device),
                ).joints[:, :22]
                joints_batches.append(joints.cpu().numpy().astype(np.float32))

        joints_np = np.concatenate(joints_batches, axis=0)
        op3_scale = compute_op3_scale_from_joints(joints_np)
        valid_clip_ranges = [(0, len(joints_np))]
        if not args.disable_feasibility_filter:
            filter_cfg = build_filter_config(args, effective_fps)
            valid_clip_ranges, local_rejections = filter_motion_clips(
                joints_np,
                effective_fps=effective_fps,
                op3_scale=op3_scale,
                cfg=filter_cfg,
            )
            rejection_reasons.update(local_rejections)
            total_candidate_clips = len(
                iter_clip_ranges(
                    num_frames=len(joints_np),
                    clip_frames=filter_cfg.clip_frames,
                    stride_frames=filter_cfg.clip_stride_frames,
                    min_frames=min(filter_cfg.clip_frames, len(joints_np)),
                )
            )
            rejected_clips += total_candidate_clips - len(valid_clip_ranges)

        if not valid_clip_ranges:
            continue

        positions, orientations, position_valid, rotation_valid, target_lin_vel_xy = build_sparse_sequence_from_joints(
            joints_np,
            effective_fps=effective_fps,
            op3_scale=op3_scale,
        )

        for clip_start, clip_end in valid_clip_ranges:
            clip = slice(clip_start, clip_end)
            position_blocks.append(positions[clip])
            orientation_blocks.append(orientations[clip])
            position_valid_blocks.append(position_valid[clip])
            rotation_valid_blocks.append(rotation_valid[clip])
            velocity_blocks.append(target_lin_vel_xy[clip])
            sequence_starts.append(total_frames)
            sequence_lengths.append(clip_end - clip_start)
            sources.append(f"{motion_path}#{clip_start}:{clip_end}")
            total_frames += clip_end - clip_start
            kept_clips += 1

    if not position_blocks:
        raise RuntimeError("No usable AMASS motion files were converted into sparse teleop commands.")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=np.concatenate(position_blocks, axis=0),
        orientations=np.concatenate(orientation_blocks, axis=0),
        position_valid=np.concatenate(position_valid_blocks, axis=0),
        rotation_valid=np.concatenate(rotation_valid_blocks, axis=0),
        target_lin_vel_xy=np.concatenate(velocity_blocks, axis=0),
        sequence_starts=np.asarray(sequence_starts, dtype=np.int64),
        sequence_lengths=np.asarray(sequence_lengths, dtype=np.int64),
        segment_names=np.asarray(SEGMENTS),
        source=np.asarray(sources, dtype=str),
        effective_fps=float(args.target_fps),
    )
    print(f"Wrote AMASS sparse dataset to: {args.output}")
    print(f"Sequences: {len(sequence_lengths)}")
    print(f"Frames: {total_frames}")
    if not args.disable_feasibility_filter:
        print(f"Kept clips: {kept_clips}")
        print(f"Rejected clips: {rejected_clips}")
        if rejection_reasons:
            print("Top rejection reasons:")
            for reason, count in rejection_reasons.most_common():
                print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
