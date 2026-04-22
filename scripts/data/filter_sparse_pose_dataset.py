#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np


DEFAULT_SEGMENTS = (
    "pelvis",
    "head",
    "left_hand",
    "right_hand",
    "left_knee",
    "right_knee",
    "left_foot",
    "right_foot",
)


@dataclass(frozen=True)
class SparseFilterConfig:
    effective_fps: float
    clip_frames: int
    clip_stride_frames: int
    min_frames: int
    max_root_speed: float
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Filter sparse OP3 teleoperation pose clips by robot feasibility.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--effective-fps", type=float, default=None)
    parser.add_argument("--clip-seconds", type=float, default=4.0)
    parser.add_argument("--filter-stride-seconds", type=float, default=4.0)
    parser.add_argument("--min-frames", type=int, default=50)
    parser.add_argument("--max-root-speed", type=float, default=0.45)
    parser.add_argument("--min-pelvis-height", type=float, default=0.22)
    parser.add_argument("--max-pelvis-height", type=float, default=0.34)
    parser.add_argument("--max-torso-lean-deg", type=float, default=35.0)
    parser.add_argument("--min-head-height", type=float, default=0.20)
    parser.add_argument("--max-foot-clearance", type=float, default=0.055)
    parser.add_argument("--max-support-foot-clearance", type=float, default=0.035)
    parser.add_argument("--min-knee-height", type=float, default=0.07)
    parser.add_argument("--max-knee-height", type=float, default=0.22)
    parser.add_argument("--max-knee-to-foot", type=float, default=0.14)
    parser.add_argument("--max-hand-distance-from-pelvis", type=float, default=0.36)
    parser.add_argument("--max-feet-separation-xy", type=float, default=0.32)
    return parser.parse_args()


def iter_clip_ranges(num_frames: int, clip_frames: int, stride_frames: int, min_frames: int) -> list[tuple[int, int]]:
    if num_frames < min_frames:
        return []
    if num_frames <= clip_frames:
        return [(0, num_frames)]

    ranges: list[tuple[int, int]] = []
    start = 0
    while start + clip_frames <= num_frames:
        ranges.append((start, start + clip_frames))
        start += stride_frames
    return ranges


def finite_rows(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values).all(axis=-1)


def resolve_effective_fps(data: np.lib.npyio.NpzFile, override: float | None) -> float:
    if override is not None:
        return float(override)
    if "effective_fps" in data:
        value = np.asarray(data["effective_fps"])
        if value.shape == ():
            return float(value)
    return 50.0


def build_config(args: argparse.Namespace, effective_fps: float) -> SparseFilterConfig:
    clip_frames = max(int(round(args.clip_seconds * effective_fps)), args.min_frames)
    clip_stride_frames = max(int(round(args.filter_stride_seconds * effective_fps)), 1)
    return SparseFilterConfig(
        effective_fps=float(effective_fps),
        clip_frames=clip_frames,
        clip_stride_frames=clip_stride_frames,
        min_frames=int(args.min_frames),
        max_root_speed=float(args.max_root_speed),
        min_pelvis_height=float(args.min_pelvis_height),
        max_pelvis_height=float(args.max_pelvis_height),
        max_torso_lean_deg=float(args.max_torso_lean_deg),
        min_head_height=float(args.min_head_height),
        max_foot_clearance=float(args.max_foot_clearance),
        max_support_foot_clearance=float(args.max_support_foot_clearance),
        min_knee_height=float(args.min_knee_height),
        max_knee_height=float(args.max_knee_height),
        max_knee_to_foot=float(args.max_knee_to_foot),
        max_hand_distance_from_pelvis=float(args.max_hand_distance_from_pelvis),
        max_feet_separation_xy=float(args.max_feet_separation_xy),
    )


def segment_index(segment_names: np.ndarray) -> dict[str, int]:
    names = [str(name) for name in segment_names.tolist()]
    missing = [name for name in DEFAULT_SEGMENTS if name not in names]
    if missing:
        raise ValueError(f"Sparse dataset is missing required segments: {missing}")
    return {name: names.index(name) for name in DEFAULT_SEGMENTS}


def source_for_sequence(data: np.lib.npyio.NpzFile, seq_index: int, input_path: Path) -> str:
    if "source" not in data:
        return f"{input_path}#{seq_index}"
    source = np.asarray(data["source"])
    if source.shape == ():
        return f"{str(source.item())}#{seq_index}"
    if len(source) > seq_index:
        return str(source[seq_index])
    return f"{input_path}#{seq_index}"


def filter_sequence(
    positions: np.ndarray,
    position_valid: np.ndarray,
    target_lin_vel_xy: np.ndarray,
    idx: dict[str, int],
    cfg: SparseFilterConfig,
) -> tuple[list[tuple[int, int]], Counter[str]]:
    pelvis = positions[:, idx["pelvis"]]
    head = positions[:, idx["head"]]
    left_hand = positions[:, idx["left_hand"]]
    right_hand = positions[:, idx["right_hand"]]
    left_knee = positions[:, idx["left_knee"]]
    right_knee = positions[:, idx["right_knee"]]
    left_foot = positions[:, idx["left_foot"]]
    right_foot = positions[:, idx["right_foot"]]

    left_foot_valid = position_valid[:, idx["left_foot"]] & finite_rows(left_foot)
    right_foot_valid = position_valid[:, idx["right_foot"]] & finite_rows(right_foot)
    foot_z_samples = np.concatenate((left_foot[left_foot_valid, 2], right_foot[right_foot_valid, 2]))
    if len(foot_z_samples) < cfg.min_frames:
        return [], Counter({"invalid_feet": 1})

    ground_z = float(np.quantile(foot_z_samples.astype(np.float32), 0.05))
    pelvis_height = pelvis[:, 2] - ground_z
    left_foot_clearance = left_foot[:, 2] - ground_z
    right_foot_clearance = right_foot[:, 2] - ground_z
    foot_clearance = np.maximum(left_foot_clearance, right_foot_clearance)
    support_foot_clearance = np.minimum(left_foot_clearance, right_foot_clearance)

    left_knee_height = left_knee[:, 2] - ground_z
    right_knee_height = right_knee[:, 2] - ground_z
    min_knee_height = np.minimum(left_knee_height, right_knee_height)
    max_knee_height = np.maximum(left_knee_height, right_knee_height)
    knee_to_foot = np.maximum(
        np.linalg.norm(left_knee - left_foot, axis=-1),
        np.linalg.norm(right_knee - right_foot, axis=-1),
    )

    torso = head - pelvis
    torso_norm = np.linalg.norm(torso, axis=-1)
    torso_cos = np.divide(torso[:, 2], torso_norm, out=np.ones_like(torso_norm), where=torso_norm > 1.0e-6)
    torso_lean_deg = np.degrees(np.arccos(np.clip(torso_cos, -1.0, 1.0)))
    head_height = head[:, 2] - pelvis[:, 2]

    hand_distance_from_pelvis = np.maximum(
        np.linalg.norm(left_hand - pelvis, axis=-1),
        np.linalg.norm(right_hand - pelvis, axis=-1),
    )
    feet_separation_xy = np.linalg.norm(left_foot[:, :2] - right_foot[:, :2], axis=-1)
    root_speed = np.linalg.norm(target_lin_vel_xy, axis=-1)

    kept: list[tuple[int, int]] = []
    rejection_counts: Counter[str] = Counter()
    clip_ranges = iter_clip_ranges(
        num_frames=len(positions),
        clip_frames=cfg.clip_frames,
        stride_frames=cfg.clip_stride_frames,
        min_frames=cfg.min_frames,
    )
    for start, end in clip_ranges:
        clip = slice(start, end)
        reasons: list[str] = []
        if float(np.max(root_speed[clip])) > cfg.max_root_speed:
            reasons.append("root_speed")
        if float(np.min(pelvis_height[clip])) < cfg.min_pelvis_height:
            reasons.append("pelvis_too_low")
        if float(np.max(pelvis_height[clip])) > cfg.max_pelvis_height:
            reasons.append("pelvis_too_high")
        if float(np.max(torso_lean_deg[clip])) > cfg.max_torso_lean_deg:
            reasons.append("torso_lean")
        if float(np.min(head_height[clip])) < cfg.min_head_height:
            reasons.append("head_too_low")
        if float(np.max(foot_clearance[clip])) > cfg.max_foot_clearance:
            reasons.append("foot_clearance")
        if float(np.max(support_foot_clearance[clip])) > cfg.max_support_foot_clearance:
            reasons.append("support_foot_float")
        if float(np.min(min_knee_height[clip])) < cfg.min_knee_height:
            reasons.append("knee_too_low")
        if float(np.max(max_knee_height[clip])) > cfg.max_knee_height:
            reasons.append("knee_too_high")
        if float(np.max(knee_to_foot[clip])) > cfg.max_knee_to_foot:
            reasons.append("knee_to_foot")
        if float(np.max(hand_distance_from_pelvis[clip])) > cfg.max_hand_distance_from_pelvis:
            reasons.append("hand_distance")
        if float(np.max(feet_separation_xy[clip])) > cfg.max_feet_separation_xy:
            reasons.append("feet_separation")
        if not np.isfinite(positions[clip]).all():
            reasons.append("non_finite")

        if reasons:
            rejection_counts.update(reasons)
        else:
            kept.append((start, end))

    return kept, rejection_counts


def main() -> None:
    args = parse_args()
    data = np.load(args.input, allow_pickle=False)

    positions = np.asarray(data["positions"], dtype=np.float32)
    orientations = np.asarray(data["orientations"], dtype=np.float32)
    position_valid = np.asarray(data["position_valid"], dtype=bool)
    rotation_valid = np.asarray(data["rotation_valid"], dtype=bool)
    target_lin_vel_xy = np.asarray(data["target_lin_vel_xy"], dtype=np.float32)
    segment_names = np.asarray(data["segment_names"] if "segment_names" in data else DEFAULT_SEGMENTS)
    idx = segment_index(segment_names)

    sequence_starts = (
        np.asarray(data["sequence_starts"], dtype=np.int64)
        if "sequence_starts" in data
        else np.asarray([0], dtype=np.int64)
    )
    sequence_lengths = (
        np.asarray(data["sequence_lengths"], dtype=np.int64)
        if "sequence_lengths" in data
        else np.asarray([len(positions)], dtype=np.int64)
    )
    effective_fps = resolve_effective_fps(data, args.effective_fps)
    cfg = build_config(args, effective_fps)

    position_blocks: list[np.ndarray] = []
    orientation_blocks: list[np.ndarray] = []
    position_valid_blocks: list[np.ndarray] = []
    rotation_valid_blocks: list[np.ndarray] = []
    velocity_blocks: list[np.ndarray] = []
    output_sequence_starts: list[int] = []
    output_sequence_lengths: list[int] = []
    output_sources: list[str] = []
    rejection_reasons: Counter[str] = Counter()
    candidate_clips = 0
    kept_clips = 0
    total_frames = 0

    for seq_idx, (seq_start, seq_len) in enumerate(zip(sequence_starts, sequence_lengths, strict=False)):
        seq_start = int(seq_start)
        seq_end = seq_start + int(seq_len)
        seq_positions = positions[seq_start:seq_end]
        seq_position_valid = position_valid[seq_start:seq_end]
        seq_vel = target_lin_vel_xy[seq_start:seq_end]
        local_ranges, local_rejections = filter_sequence(seq_positions, seq_position_valid, seq_vel, idx, cfg)
        candidate_clips += len(
            iter_clip_ranges(
                num_frames=len(seq_positions),
                clip_frames=cfg.clip_frames,
                stride_frames=cfg.clip_stride_frames,
                min_frames=cfg.min_frames,
            )
        )
        rejection_reasons.update(local_rejections)
        source = source_for_sequence(data, seq_idx, args.input)

        for local_start, local_end in local_ranges:
            clip_start = seq_start + local_start
            clip_end = seq_start + local_end
            clip = slice(clip_start, clip_end)
            position_blocks.append(positions[clip])
            orientation_blocks.append(orientations[clip])
            position_valid_blocks.append(position_valid[clip])
            rotation_valid_blocks.append(rotation_valid[clip])
            velocity_blocks.append(target_lin_vel_xy[clip])
            output_sequence_starts.append(total_frames)
            output_sequence_lengths.append(clip_end - clip_start)
            output_sources.append(f"{source}#{local_start}:{local_end}")
            total_frames += clip_end - clip_start
            kept_clips += 1

    if not position_blocks:
        raise RuntimeError("Sparse filter rejected every candidate clip.")

    output_data = {
        "positions": np.concatenate(position_blocks, axis=0),
        "orientations": np.concatenate(orientation_blocks, axis=0),
        "position_valid": np.concatenate(position_valid_blocks, axis=0),
        "rotation_valid": np.concatenate(rotation_valid_blocks, axis=0),
        "target_lin_vel_xy": np.concatenate(velocity_blocks, axis=0),
        "sequence_starts": np.asarray(output_sequence_starts, dtype=np.int64),
        "sequence_lengths": np.asarray(output_sequence_lengths, dtype=np.int64),
        "segment_names": segment_names,
        "source": np.asarray(output_sources, dtype=str),
        "effective_fps": np.asarray(effective_fps, dtype=np.float32),
        "op3_sparse_filter_config": np.asarray([json.dumps(asdict(cfg), sort_keys=True)], dtype=str),
        "op3_sparse_filter_rejections": np.asarray(
            [json.dumps(dict(rejection_reasons.most_common()), sort_keys=True)],
            dtype=str,
        ),
    }
    if "source_datasets" in data:
        output_data["source_datasets"] = np.asarray(data["source_datasets"], dtype=str)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **output_data)

    print(f"Wrote filtered sparse dataset to: {args.output}")
    print(f"Kept clips: {kept_clips}")
    print(f"Rejected clips: {candidate_clips - kept_clips}")
    print(f"Frames: {total_frames}")
    if rejection_reasons:
        print("Top sparse rejection reasons:")
        for reason, count in rejection_reasons.most_common():
            print(f"  - {reason}: {count}")


if __name__ == "__main__":
    main()
