#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Play an OP3 ADD checkpoint.")
    parser.add_argument("--task", default="Isaac-OP3-Teleop-Newton-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--teleop_mode", choices=("synthetic", "dataset"), default=None)
    parser.add_argument("--teleop_dataset_path", default=None)
    parser.add_argument(
        "--config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "source/op3_teleop_lab/op3_teleop_lab/tasks/direct/op3_teleop/agents/add_ppo_cfg.yaml"
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    return parser


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
    import torch

    import op3_teleop_lab.tasks  # noqa: F401
    from op3_teleop_lab.learning.add.config import ADDTrainingConfig
    from op3_teleop_lab.learning.add.trainer import ADDTrainer
    from op3_teleop_lab.tasks.direct.op3_teleop.env_cfg import OP3TeleopEnvCfg, OP3TeleopNewtonEnvCfg

    cfg = OP3TeleopNewtonEnvCfg() if "Newton" in args.task else OP3TeleopEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.teleop_mode is not None:
        cfg.teleop_mode = args.teleop_mode
    if args.teleop_dataset_path is not None:
        cfg.teleop_dataset_path = args.teleop_dataset_path

    train_cfg = ADDTrainingConfig.from_yaml(args.config)
    env = gym.make(args.task, cfg=cfg)
    base_env = env.unwrapped
    obs_dict, extras = env.reset()
    obs = obs_dict["policy"]
    diff = extras.get("add_diff")
    if diff is None:
        zero_action = torch.zeros((base_env.num_envs, env.action_space.shape[-1]), device=obs.device)
        obs_dict, _, _, _, extras = env.step(zero_action)
        obs = obs_dict["policy"]
        diff = extras["add_diff"]

    trainer = ADDTrainer(
        env=base_env,
        obs_dim=obs.shape[-1],
        action_dim=env.action_space.shape[-1],
        diff_dim=diff.shape[-1],
        config=train_cfg,
        device=obs.device,
        out_dir=Path(args.checkpoint).resolve().parent,
    )
    trainer.load(args.checkpoint)

    obs = trainer.obs
    with torch.no_grad():
        for _ in range(args.steps):
            actions = trainer.policy.deterministic(obs)
            obs_dict, _, _, _, _ = env.step(actions)
            obs = obs_dict["policy"]

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
