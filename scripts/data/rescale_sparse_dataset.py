#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from embodiment_profiles import get_embodiment_profile


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Rescale sparse teleop positions/velocities between embodiments.")
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source-embodiment", default="op3")
    parser.add_argument("--target-embodiment", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_profile = get_embodiment_profile(args.source_embodiment)
    target_profile = get_embodiment_profile(args.target_embodiment)
    scale = float(target_profile.target_body_scale_m / source_profile.target_body_scale_m)

    with np.load(args.input, allow_pickle=False) as data:
        if "embodiment" not in data:
            raise KeyError(f"Input sparse dataset {args.input} does not declare an embodiment.")
        input_embodiment = str(np.asarray(data["embodiment"]).item()).strip().lower()
        if input_embodiment != source_profile.name:
            raise ValueError(
                f"Input sparse dataset embodiment mismatch: expected {source_profile.name!r}, "
                f"found {input_embodiment!r}."
            )
        payload: dict[str, np.ndarray] = {key: data[key] for key in data.files}

    payload["positions"] = payload["positions"].astype(np.float32) * np.float32(scale)
    if "target_lin_vel_xy" in payload:
        payload["target_lin_vel_xy"] = payload["target_lin_vel_xy"].astype(np.float32) * np.float32(scale)
    payload["embodiment"] = np.asarray(target_profile.name, dtype=str)
    payload["rescale_source_embodiment"] = np.asarray(source_profile.name, dtype=str)
    payload["rescale_factor"] = np.asarray(scale, dtype=np.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload)
    print(f"Rescaled sparse dataset written to: {args.output}")
    print(f"Source embodiment: {source_profile.name}")
    print(f"Target embodiment: {target_profile.name}")
    print(f"Scale factor: {scale:.6f}")


if __name__ == "__main__":
    main()
