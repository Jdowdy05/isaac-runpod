#!/usr/bin/env python3

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np


AIST_KEYPOINTS = {
    "nose": 0,
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


def build_sparse_sequence(keypoints3d: np.ndarray, effective_fps: float) -> tuple[np.ndarray, ...]:
    num_frames = keypoints3d.shape[0]
    positions = np.zeros((num_frames, len(SEGMENTS), 3), dtype=np.float32)
    orientations = np.zeros((num_frames, len(SEGMENTS), 4), dtype=np.float32)
    orientations[..., 0] = 1.0
    position_valid = np.zeros((num_frames, len(SEGMENTS)), dtype=bool)
    rotation_valid = np.zeros((num_frames, len(SEGMENTS)), dtype=bool)

    left_hip = keypoints3d[:, AIST_KEYPOINTS["left_hip"]]
    right_hip = keypoints3d[:, AIST_KEYPOINTS["right_hip"]]
    pelvis = 0.5 * (left_hip + right_hip)

    raw_targets = {
        "pelvis": pelvis,
        "head": keypoints3d[:, AIST_KEYPOINTS["nose"]],
        "left_hand": keypoints3d[:, AIST_KEYPOINTS["left_wrist"]],
        "right_hand": keypoints3d[:, AIST_KEYPOINTS["right_wrist"]],
        "left_knee": keypoints3d[:, AIST_KEYPOINTS["left_knee"]],
        "right_knee": keypoints3d[:, AIST_KEYPOINTS["right_knee"]],
        "left_foot": keypoints3d[:, AIST_KEYPOINTS["left_ankle"]],
        "right_foot": keypoints3d[:, AIST_KEYPOINTS["right_ankle"]],
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
    speed = np.linalg.norm(pelvis_vel_xy, axis=-1, keepdims=True).astype(np.float32)
    target_lin_vel_xy = np.concatenate((speed, np.zeros_like(speed)), axis=-1)

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
