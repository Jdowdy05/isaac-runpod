#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from common import (
    create_rsl_runner,
    default_dataset_path,
    format_action_stats,
    format_joint_target_debug,
    policy_obs_tensor,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Play an OP3 RSL or RSL-ADD checkpoint.")
    parser.add_argument("--task", default="Isaac-OP3-Teleop-Direct-v0")
    parser.add_argument("--runner", choices=("ppo", "add"), default="ppo")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--teleop_mode", choices=("synthetic", "dataset"), default=None)
    parser.add_argument("--teleop_dataset_path", default=None)
    parser.add_argument(
        "--add_config",
        default=str(
            Path(__file__).resolve().parents[2]
            / "source/op3_teleop_lab/op3_teleop_lab/tasks/direct/op3_teleop/agents/add_ppo_cfg.yaml"
        ),
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument(
        "--sample_actions",
        action="store_true",
        help="Sample from the stochastic policy distribution instead of taking deterministic means.",
    )
    parser.add_argument("--print_stats_every", type=int, default=0, help="Print observation and action statistics every N steps.")
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
    from op3_teleop_lab.tasks.direct.op3_teleop.env_cfg import OP3TeleopEnvCfg, OP3TeleopNewtonEnvCfg

    if args.teleop_mode is None and args.teleop_dataset_path is None:
        args.teleop_mode, args.teleop_dataset_path = default_dataset_path()

    cfg = OP3TeleopNewtonEnvCfg() if "Newton" in args.task else OP3TeleopEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.teleop_mode is not None:
        cfg.teleop_mode = args.teleop_mode
    if args.teleop_dataset_path is not None:
        cfg.teleop_dataset_path = args.teleop_dataset_path
    if args.runner == "add":
        cfg.add_diff_reward_weight = 0.0

    env = gym.make(args.task, cfg=cfg)
    base_env = env.unwrapped
    obs_dict, extras = env.reset()
    actor_obs = obs_dict["policy"]
    diff = extras.get("add_diff")
    if args.runner == "add" and diff is None:
        zero_action = torch.zeros((base_env.num_envs, env.action_space.shape[-1]), device=actor_obs.device)
        obs_dict, _, _, _, extras = env.step(zero_action)
        diff = extras["add_diff"]

    wrapped_env, runner, runner_device = create_rsl_runner(
        env=env,
        runner_kind=args.runner,
        add_config_path=args.add_config if args.runner == "add" else None,
        diff_dim=None if diff is None else diff.shape[-1],
        checkpoint_dir=Path(args.checkpoint).resolve().parent,
        device=getattr(base_env.cfg.sim, "device", None),
    )
    runner.load(args.checkpoint, map_location=runner_device)
    policy = runner.get_inference_policy(device=runner_device)
    obs = wrapped_env.get_observations().to(runner_device)
    action_names = list(getattr(base_env, "action_joint_names", base_env.cfg.profile.joint_names))

    with torch.no_grad():
        for step in range(args.steps):
            actions = policy(obs, stochastic_output=args.sample_actions)

            if args.print_stats_every > 0 and step % args.print_stats_every == 0:
                obs_tensor = policy_obs_tensor(obs)
                action_abs = actions.abs()
                print(
                    f"[play step {step}] obs_abs_mean={obs_tensor.abs().mean().item():.6f} "
                    f"obs_abs_max={obs_tensor.abs().max().item():.6f} "
                    f"action_abs_mean={action_abs.mean().item():.6f} "
                    f"action_abs_max={action_abs.max().item():.6f}"
                )
                print(f"[play step {step}] actions_env0: {format_action_stats(action_names, actions)}")
                print(f"[play step {step}] target_debug_env0: {format_joint_target_debug(base_env, action_names, actions)}")

            obs, _, _, _ = wrapped_env.step(actions)

    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
