#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


REQUIRED_KEYS = (
    "positions",
    "orientations",
    "position_valid",
    "rotation_valid",
    "target_lin_vel_xy",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Merge one or more sparse teleoperation datasets into a single NPZ.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    positions: list[np.ndarray] = []
    orientations: list[np.ndarray] = []
    position_valid: list[np.ndarray] = []
    rotation_valid: list[np.ndarray] = []
    target_lin_vel_xy: list[np.ndarray] = []
    sequence_starts: list[int] = []
    sequence_lengths: list[int] = []
    sources: list[str] = []
    total_frames = 0

    segment_names: np.ndarray | None = None

    for input_path in args.inputs:
        if not input_path.exists():
            raise FileNotFoundError(f"Input sparse dataset does not exist: {input_path}")

        data = np.load(input_path, allow_pickle=False)
        for key in REQUIRED_KEYS:
            if key not in data:
                raise KeyError(f"Dataset {input_path} is missing required key: {key}")

        if segment_names is None:
            segment_names = data["segment_names"] if "segment_names" in data else None
        elif "segment_names" in data and not np.array_equal(segment_names, data["segment_names"]):
            raise ValueError(f"Segment names do not match across datasets; failed on {input_path}")

        positions.append(np.asarray(data["positions"], dtype=np.float32))
        orientations.append(np.asarray(data["orientations"], dtype=np.float32))
        position_valid.append(np.asarray(data["position_valid"], dtype=bool))
        rotation_valid.append(np.asarray(data["rotation_valid"], dtype=bool))
        target_lin_vel_xy.append(np.asarray(data["target_lin_vel_xy"], dtype=np.float32))

        local_lengths = (
            np.asarray(data["sequence_lengths"], dtype=np.int64)
            if "sequence_lengths" in data
            else np.asarray([len(data["positions"])], dtype=np.int64)
        )
        local_starts = (
            np.asarray(data["sequence_starts"], dtype=np.int64)
            if "sequence_starts" in data
            else np.asarray([0], dtype=np.int64)
        )

        sequence_starts.extend((local_starts + total_frames).tolist())
        sequence_lengths.extend(local_lengths.tolist())
        total_frames += int(len(data["positions"]))
        sources.append(str(input_path))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        positions=np.concatenate(positions, axis=0),
        orientations=np.concatenate(orientations, axis=0),
        position_valid=np.concatenate(position_valid, axis=0),
        rotation_valid=np.concatenate(rotation_valid, axis=0),
        target_lin_vel_xy=np.concatenate(target_lin_vel_xy, axis=0),
        sequence_starts=np.asarray(sequence_starts, dtype=np.int64),
        sequence_lengths=np.asarray(sequence_lengths, dtype=np.int64),
        segment_names=segment_names if segment_names is not None else np.asarray([], dtype=str),
        source_datasets=np.asarray(sources, dtype=str),
    )
    print(f"Wrote merged sparse dataset to: {args.output}")
    print(f"Frames: {total_frames}")
    print(f"Inputs: {len(sources)}")


if __name__ == "__main__":
    main()
