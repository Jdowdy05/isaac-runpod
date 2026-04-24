#!/usr/bin/env python3

from __future__ import annotations

import argparse
import random
from datetime import datetime
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train OP3 teleoperation with RSL-RL PPO plus online ADD.")
    parser.add_argument("--task", default="Isaac-OP3-Teleop-Direct-v0")
    parser.add_argument("--num_envs", type=int, default=2048)
    parser.add_argument("--teleop_mode", choices=("synthetic", "dataset"), default=None)
    parser.add_argument("--teleop_dataset_path", default=None)
    parser.add_argument(
        "--add_config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "source/op3_teleop_lab/op3_teleop_lab/tasks/direct/op3_teleop/agents/add_ppo_cfg.yaml"
        ),
    )
    parser.add_argument("--out_dir", default=str(Path(__file__).resolve().parents[2] / "logs/rsl_add/op3_teleop"))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--max_iterations", type=int, default=None)
    parser.add_argument(
        "--keep_env_add_diff_reward",
        action="store_true",
        help="Keep the dense env ADD-style reward. By default RSL-ADD disables it and uses discriminator reward.",
    )
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
    import numpy as np
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

    import op3_teleop_lab.tasks  # noqa: F401
    from op3_teleop_lab.learning.add.config import ADDTrainingConfig
    from op3_teleop_lab.learning.rsl_add import RslAddOnPolicyRunner
    from op3_teleop_lab.tasks.direct.op3_teleop.agents.rsl_rl_ppo_cfg import OP3TeleopPPORunnerCfg
    from op3_teleop_lab.tasks.direct.op3_teleop.env_cfg import OP3TeleopEnvCfg, OP3TeleopNewtonEnvCfg

    env_cfg = OP3TeleopNewtonEnvCfg() if "Newton" in args.task else OP3TeleopEnvCfg()
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.sim.device = args.device if getattr(args, "device", None) is not None else env_cfg.sim.device
    if args.teleop_mode is not None:
        env_cfg.teleop_mode = args.teleop_mode
    if args.teleop_dataset_path is not None:
        env_cfg.teleop_dataset_path = args.teleop_dataset_path
    if not args.keep_env_add_diff_reward:
        env_cfg.add_diff_reward_weight = 0.0

    agent_cfg = OP3TeleopPPORunnerCfg()
    if getattr(args, "device", None) is not None:
        agent_cfg.device = args.device
    if args.max_iterations is not None:
        agent_cfg.max_iterations = args.max_iterations

    add_cfg = ADDTrainingConfig.from_yaml(args.add_config)
    if args.max_iterations is not None:
        add_cfg.max_iterations = args.max_iterations

    random.seed(add_cfg.seed)
    np.random.seed(add_cfg.seed)
    torch.manual_seed(add_cfg.seed)
    env_cfg.seed = add_cfg.seed
    agent_cfg.seed = add_cfg.seed

    log_root = Path(args.out_dir).resolve()
    log_dir = Path(args.checkpoint).resolve().parent if args.checkpoint else log_root / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    print(f"RSL-ADD log/checkpoint directory: {log_dir}", flush=True)

    env = gym.make(args.task, cfg=env_cfg)
    obs_dict, extras = env.reset()
    actor_obs = obs_dict["policy"]
    diff = extras.get("add_diff")
    if diff is None:
        with torch.no_grad():
            zero_action = torch.zeros((env.unwrapped.num_envs, env.action_space.shape[-1]), device=actor_obs.device)
            obs_dict, _, _, _, extras = env.step(zero_action)
            diff = extras["add_diff"]
    diff_dim = diff.shape[-1]

    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner = RslAddOnPolicyRunner(
        wrapped_env,
        agent_cfg.to_dict(),
        add_cfg=add_cfg,
        diff_dim=diff_dim,
        log_dir=log_dir,
        device=agent_cfg.device,
    )
    if args.checkpoint:
        runner.load(args.checkpoint)
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
