#!/usr/bin/env python3

from __future__ import annotations

import gymnasium as gym

import op3_teleop_lab.tasks  # noqa: F401


def main() -> None:
    env_ids = sorted(
        spec.id
        for spec in gym.registry.values()
        if spec.id.startswith("Isaac-") and "-Teleop-" in spec.id
    )
    if not env_ids:
        print("No teleop environments found.")
        return

    print("Registered teleop environments:")
    for env_id in env_ids:
        print(f"  - {env_id}")


if __name__ == "__main__":
    main()
