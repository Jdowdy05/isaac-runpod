#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from datetime import date
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Train sparse humanoid teleoperation with ADD.")
    parser.add_argument("--task", default="Isaac-OP3-Teleop-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--teleop_mode", choices=("synthetic", "dataset"), default=None)
    parser.add_argument("--teleop_dataset_path", default=None)
    parser.add_argument(
        "--config",
        default=None,
    )
    parser.add_argument("--out_dir", default=None)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    return parser


def resolve_session_out_dir(base_out_dir: str | Path, checkpoint: str | None) -> Path:
    if checkpoint:
        return Path(checkpoint).resolve().parent

    root = Path(base_out_dir).resolve()
    date_stem = date.today().isoformat()
    candidate = root / date_stem
    if not candidate.exists():
        return candidate

    suffix = 2
    while True:
        candidate = root / f"{date_stem}_{suffix:02d}"
        if not candidate.exists():
            return candidate
        suffix += 1


def main() -> None:
    parser = build_arg_parser()
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # type: ignore[no-redef]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import numpy as np
    import torch

    import op3_teleop_lab.tasks  # noqa: F401
    from op3_teleop_lab.learning.add.config import ADDTrainingConfig
    from op3_teleop_lab.learning.add.trainer import ADDTrainer
    from op3_teleop_lab.tasks.task_registry import (
        make_env_cfg_for_task,
        resolve_add_config_path_for_task,
        task_slug_for_task,
    )

    cfg = make_env_cfg_for_task(args.task)
    cfg.scene.num_envs = args.num_envs
    if args.teleop_mode is not None:
        cfg.teleop_mode = args.teleop_mode
    if args.teleop_dataset_path is not None:
        cfg.teleop_dataset_path = args.teleop_dataset_path

    config_path = args.config if args.config is not None else resolve_add_config_path_for_task(args.task)
    train_cfg = ADDTrainingConfig.from_yaml(config_path)
    if args.max_iterations is not None:
        train_cfg.max_iterations = args.max_iterations
    base_out_dir = (
        Path(args.out_dir)
        if args.out_dir is not None
        else Path(__file__).resolve().parents[2] / "checkpoints" / "add" / task_slug_for_task(args.task)
    )
    session_out_dir = resolve_session_out_dir(base_out_dir, args.checkpoint)
    print(f"Checkpoint directory: {session_out_dir}")

    random.seed(train_cfg.seed)
    np.random.seed(train_cfg.seed)
    torch.manual_seed(train_cfg.seed)

    env = gym.make(args.task, cfg=cfg)
    base_env = env.unwrapped
    obs_dict, extras = env.reset()
    actor_obs = obs_dict["policy"]
    critic_obs = obs_dict.get("critic", actor_obs)
    diff = extras.get("add_diff")
    if diff is None:
        # first reset can return empty extras before the first env step
        with torch.no_grad():
            zero_action = torch.zeros((base_env.num_envs, env.action_space.shape[-1]), device=actor_obs.device)
            obs_dict, _, _, _, extras = env.step(zero_action)
            actor_obs = obs_dict["policy"]
            critic_obs = obs_dict.get("critic", actor_obs)
            diff = extras["add_diff"]

    trainer = ADDTrainer(
        env=base_env,
        obs_dim=actor_obs.shape[-1],
        action_dim=env.action_space.shape[-1],
        diff_dim=diff.shape[-1],
        config=train_cfg,
        device=actor_obs.device,
        out_dir=session_out_dir,
        critic_obs_dim=critic_obs.shape[-1],
    )
    if args.checkpoint:
        trainer.load(args.checkpoint)
    trainer.train(num_iterations=args.max_iterations)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
