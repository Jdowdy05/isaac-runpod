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
    "embodiment",
)


def _read_scalar_string(data: np.lib.npyio.NpzFile, key: str, input_path: Path) -> str:
    value = np.asarray(data[key])
    if value.shape != ():
        raise ValueError(f"Dataset {input_path} key {key!r} must be a scalar string.")
    return str(value.item()).strip().lower()


def _resolve_sequence_fps(
    data: np.lib.npyio.NpzFile,
    sequence_count: int,
    input_path: Path,
) -> np.ndarray:
    if "sequence_fps" in data:
        value = np.asarray(data["sequence_fps"], dtype=np.float32)
    elif "effective_fps" in data:
        value = np.asarray(data["effective_fps"], dtype=np.float32)
    else:
        raise KeyError(f"Dataset {input_path} is missing required FPS metadata: sequence_fps or effective_fps")

    if value.shape == ():
        fps = np.full(sequence_count, float(value), dtype=np.float32)
    elif value.ndim == 1 and len(value) == sequence_count:
        fps = value.astype(np.float32)
    else:
        raise ValueError(
            f"Dataset {input_path} FPS metadata must be scalar or length {sequence_count}, got shape {value.shape}."
        )
    if not np.isfinite(fps).all() or np.any(fps <= 0.0):
        raise ValueError(f"Dataset {input_path} contains invalid FPS values.")
    return fps


def _effective_fps_metadata(sequence_fps: np.ndarray) -> np.ndarray:
    if len(sequence_fps) == 0:
        return np.asarray(0.0, dtype=np.float32)
    if np.allclose(sequence_fps, sequence_fps[0]):
        return np.asarray(float(sequence_fps[0]), dtype=np.float32)
    return sequence_fps.astype(np.float32)


def _validate_sequence_ranges(
    starts: np.ndarray,
    lengths: np.ndarray,
    total_frames: int,
    input_path: Path,
) -> None:
    if starts.ndim != 1 or lengths.ndim != 1 or starts.shape != lengths.shape:
        raise ValueError(
            f"Dataset {input_path} sequence_starts and sequence_lengths must be matching 1-D arrays; "
            f"got {starts.shape} and {lengths.shape}."
        )
    if np.any(lengths <= 0):
        raise ValueError(f"Dataset {input_path} sequence_lengths must be strictly positive.")
    if np.any(starts < 0) or np.any(starts + lengths > total_frames):
        raise ValueError(f"Dataset {input_path} sequence ranges are outside the frame arrays.")


def _resolve_sequence_sources(
    data: np.lib.npyio.NpzFile,
    sequence_count: int,
    input_path: Path,
) -> list[str]:
    if "sequence_source_dataset" not in data:
        return [str(input_path)] * sequence_count
    sources = np.asarray(data["sequence_source_dataset"], dtype=str)
    if sources.ndim != 1 or len(sources) != sequence_count:
        raise ValueError(
            f"Dataset {input_path} sequence_source_dataset must be length {sequence_count}, got shape {sources.shape}."
        )
    return [str(source) for source in sources.tolist()]


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
    sequence_fps: list[float] = []
    sequence_source_dataset: list[str] = []
    sources: list[str] = []
    total_frames = 0

    segment_names: np.ndarray | None = None
    embodiment: str | None = None

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

        local_embodiment = _read_scalar_string(data, "embodiment", input_path)
        if embodiment is None:
            embodiment = local_embodiment
        elif embodiment != local_embodiment:
            raise ValueError(
                f"Embodiment mismatch across sparse datasets: expected {embodiment!r}, "
                f"found {local_embodiment!r} in {input_path}."
            )

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
        _validate_sequence_ranges(local_starts, local_lengths, int(len(data["positions"])), input_path)

        sequence_starts.extend((local_starts + total_frames).tolist())
        sequence_lengths.extend(local_lengths.tolist())
        sequence_fps.extend(_resolve_sequence_fps(data, len(local_lengths), input_path).tolist())
        sequence_source_dataset.extend(_resolve_sequence_sources(data, len(local_lengths), input_path))
        total_frames += int(len(data["positions"]))
        sources.append(str(input_path))

    output_sequence_fps = np.asarray(sequence_fps, dtype=np.float32)
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
        sequence_fps=output_sequence_fps,
        sequence_source_dataset=np.asarray(sequence_source_dataset, dtype=str),
        effective_fps=_effective_fps_metadata(output_sequence_fps),
        segment_names=segment_names if segment_names is not None else np.asarray([], dtype=str),
        source_datasets=np.asarray(sources, dtype=str),
        embodiment=np.asarray(embodiment, dtype=str),
    )
    print(f"Wrote merged sparse dataset to: {args.output}")
    print(f"Frames: {total_frames}")
    print(f"Inputs: {len(sources)}")


if __name__ == "__main__":
    main()
