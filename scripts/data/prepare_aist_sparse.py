#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


AIST_KEYPOINTS = {
    "nose": 0,
    "left_shoulder": 5,
    "right_shoulder": 6,
    "left_elbow": 7,
    "right_elbow": 8,
    "left_wrist": 9,
    "right_wrist": 10,
    "left_hip": 11,
    "right_hip": 12,
    "left_knee": 13,
    "right_knee": 14,
    "left_ankle": 15,
    "right_ankle": 16,
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert AIST++ keypoints into sparse-pose teleop commands.")
    parser.add_argument("--aist-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stride", type=int, default=2, help="Subsample factor from 60 Hz.")
    parser.add_argument("--min-frames", type=int, default=60)
    parser.add_argument("--limit", type=int, default=None)
    return parser.parse_args()


def load_ignore_set(aist_root: Path) -> set[str]:
    ignore_path = aist_root / "ignore_list.txt"
    if not ignore_path.exists():
        return set()
    return {line.strip() for line in ignore_path.read_text().splitlines() if line.strip()}


def load_keypoints3d(path: Path) -> np.ndarray:
    with path.open("rb") as f:
        try:
            data = pickle.load(f)
        except TypeError:
            f.seek(0)
            data = pickle.load(f, encoding="latin1")
    key = "keypoints3d_optim" if "keypoints3d_optim" in data else "keypoints3d"
    return np.asarray(data[key], dtype=np.float32)


def finite_mask(points: np.ndarray) -> np.ndarray:
    return np.isfinite(points).all(axis=-1)


def forward_fill(points: np.ndarray) -> np.ndarray:
    output = points.copy()
    valid = np.isfinite(output).all(axis=-1)
    if not np.any(valid):
        return np.zeros_like(output)
    first_valid = int(np.argmax(valid))
    output[:first_valid] = output[first_valid]
    for i in range(first_valid + 1, len(output)):
        if not valid[i]:
            output[i] = output[i - 1]
    return output


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


def build_sparse_sequence(keypoints3d: np.ndarray, effective_fps: float) -> tuple[np.ndarray, ...]:
    num_frames = keypoints3d.shape[0]
    positions = np.zeros((num_frames, len(SEGMENTS), 3), dtype=np.float32)
    orientations = np.zeros((num_frames, len(SEGMENTS), 4), dtype=np.float32)
    orientations[..., 3] = 1.0
    position_valid = np.zeros((num_frames, len(SEGMENTS)), dtype=bool)
    rotation_valid = np.zeros((num_frames, len(SEGMENTS)), dtype=bool)

    left_shoulder = keypoints3d[:, AIST_KEYPOINTS["left_shoulder"]]
    right_shoulder = keypoints3d[:, AIST_KEYPOINTS["right_shoulder"]]
    left_elbow = keypoints3d[:, AIST_KEYPOINTS["left_elbow"]]
    right_elbow = keypoints3d[:, AIST_KEYPOINTS["right_elbow"]]
    left_hip = keypoints3d[:, AIST_KEYPOINTS["left_hip"]]
    right_hip = keypoints3d[:, AIST_KEYPOINTS["right_hip"]]
    left_knee = keypoints3d[:, AIST_KEYPOINTS["left_knee"]]
    right_knee = keypoints3d[:, AIST_KEYPOINTS["right_knee"]]
    left_ankle = keypoints3d[:, AIST_KEYPOINTS["left_ankle"]]
    right_ankle = keypoints3d[:, AIST_KEYPOINTS["right_ankle"]]
    nose = keypoints3d[:, AIST_KEYPOINTS["nose"]]

    pelvis = 0.5 * (left_hip + right_hip)
    shoulder_center = 0.5 * (left_shoulder + right_shoulder)

    raw_targets = {
        "pelvis": pelvis,
        "head": nose,
        "left_hand": keypoints3d[:, AIST_KEYPOINTS["left_wrist"]],
        "right_hand": keypoints3d[:, AIST_KEYPOINTS["right_wrist"]],
        "left_knee": left_knee,
        "right_knee": right_knee,
        "left_foot": left_ankle,
        "right_foot": right_ankle,
    }

    pelvis_valid = finite_mask(pelvis)
    pelvis_filled = forward_fill(pelvis)

    for segment_name, points in raw_targets.items():
        seg_idx = SEGMENT_INDEX[segment_name]
        valid = finite_mask(points) & pelvis_valid
        positions[:, seg_idx] = points - pelvis_filled
        positions[:, seg_idx][~valid] = 0.0
        position_valid[:, seg_idx] = valid

    positions[:, SEGMENT_INDEX["pelvis"]] = 0.0
    position_valid[:, SEGMENT_INDEX["pelvis"]] = pelvis_valid

    pelvis_vel_xy = np.diff(pelvis_filled[:, :2], axis=0, prepend=pelvis_filled[:1, :2]) * effective_fps
    target_lin_vel_xy = pelvis_vel_xy.astype(np.float32)

    pelvis_lateral = (left_hip - right_hip) + (left_shoulder - right_shoulder)
    pelvis_up_hint = shoulder_center - pelvis
    pelvis_forward_hint = np.cross(pelvis_lateral, pelvis_up_hint)
    pelvis_world, pelvis_rot_valid = make_frame_from_forward_up(pelvis_forward_hint, pelvis_up_hint)
    pelvis_world_inv = np.swapaxes(pelvis_world, -1, -2)
    pelvis_forward = pelvis_world[:, :, 0]
    pelvis_up = pelvis_world[:, :, 2]

    orientations[:, SEGMENT_INDEX["pelvis"]] = rotation_matrices_to_quats_xyzw(pelvis_world)
    rotation_valid[:, SEGMENT_INDEX["pelvis"]] = pelvis_rot_valid

    def set_relative_orientation(
        segment_name: str,
        world_mat: np.ndarray,
        valid_mask: np.ndarray,
    ) -> None:
        seg_idx = SEGMENT_INDEX[segment_name]
        relative_mat = np.einsum("tij,tjk->tik", pelvis_world_inv, world_mat).astype(np.float32)
        orientations[:, seg_idx] = rotation_matrices_to_quats_xyzw(relative_mat)
        rotation_valid[:, seg_idx] = valid_mask & pelvis_rot_valid

    head_forward_hint = np.cross(left_shoulder - right_shoulder, nose - shoulder_center)
    head_world, head_rot_valid = make_frame_from_forward_up(head_forward_hint, nose - shoulder_center)
    set_relative_orientation("head", head_world, head_rot_valid)

    left_hand_world, left_hand_rot_valid = make_frame_from_forward_up(
        raw_targets["left_hand"] - left_elbow,
        pelvis_up,
    )
    set_relative_orientation("left_hand", left_hand_world, left_hand_rot_valid)

    right_hand_world, right_hand_rot_valid = make_frame_from_forward_up(
        raw_targets["right_hand"] - right_elbow,
        pelvis_up,
    )
    set_relative_orientation("right_hand", right_hand_world, right_hand_rot_valid)

    left_knee_world, left_knee_rot_valid = make_frame_from_forward_up(left_ankle - left_knee, pelvis_up)
    set_relative_orientation("left_knee", left_knee_world, left_knee_rot_valid)

    right_knee_world, right_knee_rot_valid = make_frame_from_forward_up(right_ankle - right_knee, pelvis_up)
    set_relative_orientation("right_knee", right_knee_world, right_knee_rot_valid)

    left_foot_world, left_foot_rot_valid = make_frame_from_forward_up(pelvis_forward, left_knee - left_ankle)
    set_relative_orientation("left_foot", left_foot_world, left_foot_rot_valid)

    right_foot_world, right_foot_rot_valid = make_frame_from_forward_up(pelvis_forward, right_knee - right_ankle)
    set_relative_orientation("right_foot", right_foot_world, right_foot_rot_valid)

    return positions, orientations, position_valid, rotation_valid, target_lin_vel_xy


def main() -> None:
    args = parse_args()
    keypoints_root = args.aist_root / "keypoints3d"
    if not keypoints_root.exists():
        raise FileNotFoundError(f"Expected AIST++ keypoints3d directory at: {keypoints_root}")

    effective_fps = 60.0 / float(args.stride)
    ignore_set = load_ignore_set(args.aist_root)

    position_blocks: list[np.ndarray] = []
    orientation_blocks: list[np.ndarray] = []
    position_valid_blocks: list[np.ndarray] = []
    rotation_valid_blocks: list[np.ndarray] = []
    velocity_blocks: list[np.ndarray] = []
    sequence_starts: list[int] = []
    sequence_lengths: list[int] = []
    total_frames = 0

    sequence_paths = sorted(keypoints_root.glob("*.pkl"))
    if args.limit is not None:
        sequence_paths = sequence_paths[: args.limit]

    for sequence_path in sequence_paths:
        sequence_name = sequence_path.stem
        if sequence_name in ignore_set:
            continue

        keypoints3d = load_keypoints3d(sequence_path)[:: args.stride]
        if len(keypoints3d) < args.min_frames:
            continue

        positions, orientations, position_valid, rotation_valid, target_lin_vel_xy = build_sparse_sequence(
            keypoints3d,
            effective_fps=effective_fps,
        )

        position_blocks.append(positions)
        orientation_blocks.append(orientations)
        position_valid_blocks.append(position_valid)
        rotation_valid_blocks.append(rotation_valid)
        velocity_blocks.append(target_lin_vel_xy)
        sequence_starts.append(total_frames)
        sequence_lengths.append(len(positions))
        total_frames += len(positions)

    if not position_blocks:
        raise RuntimeError("No usable AIST++ sequences were found.")

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
        source="AIST++ keypoints3d",
        effective_fps=effective_fps,
    )

    print(f"Wrote sparse AIST dataset to: {args.output}")
    print(f"Sequences: {len(sequence_starts)}")
    print(f"Frames: {total_frames}")


if __name__ == "__main__":
    main()
