#!/usr/bin/env python3

from __future__ import annotations

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path


def find_isaaclab_root() -> Path:
    candidates = []
    if os.environ.get("ISAACLAB_ROOT"):
        candidates.append(Path(os.environ["ISAACLAB_ROOT"]).expanduser())
    candidates.extend(
        (
            Path("/workspace/IsaacLab"),
            Path(__file__).resolve().parents[2] / "IsaacLab",
            Path.cwd() / "IsaacLab",
        )
    )
    for candidate in candidates:
        if (candidate / "isaaclab.sh").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate Isaac Lab. Set ISAACLAB_ROOT or clone IsaacLab into /workspace/IsaacLab."
    )


def parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description="Wrapper around Isaac Lab RL Games inference.")
    parser.add_argument("--task", required=True, help="Registered gym task id.")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--num_envs", type=int)
    parser.add_argument("--teleop_mode", choices=("synthetic", "dataset"), default=None)
    parser.add_argument("--teleop_dataset_path", default=None)
    return parser.parse_known_args()


def main() -> None:
    args, passthrough = parse_args()
    isaaclab_root = find_isaaclab_root()
    play_script = isaaclab_root / "scripts" / "reinforcement_learning" / "rl_games" / "play.py"
    if not play_script.exists():
        raise FileNotFoundError(f"Could not find Isaac Lab RL Games play script: {play_script}")

    env = os.environ.copy()
    if args.teleop_mode:
        env["HUMANOID_TELEOP_MODE"] = args.teleop_mode
        env["OP3_TELEOP_MODE"] = args.teleop_mode
    if args.teleop_dataset_path:
        env["HUMANOID_TELEOP_DATASET_PATH"] = args.teleop_dataset_path
        env["OP3_TELEOP_DATASET_PATH"] = args.teleop_dataset_path

    cmd = [sys.executable, str(play_script), "--task", args.task, "--checkpoint", args.checkpoint]
    if args.num_envs is not None:
        cmd.extend(("--num_envs", str(args.num_envs)))
    cmd.extend(passthrough)

    print(shlex.join(cmd))
    subprocess.run(cmd, cwd=isaaclab_root, env=env, check=True)


if __name__ == "__main__":
    main()
