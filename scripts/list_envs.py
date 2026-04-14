#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym

import op3_teleop_lab.tasks  # noqa: F401


def main() -> None:
    env_ids = sorted(spec.id for spec in gym.registry.values() if "OP3-Teleop" in spec.id)
    if not env_ids:
        print("No OP3 teleop environments found.")
        return

    print("Registered OP3 teleop environments:")
    for env_id in env_ids:
        print(f"  - {env_id}")


if __name__ == "__main__":
    main()

