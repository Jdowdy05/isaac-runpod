#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Play an OP3 ADD checkpoint.")
    parser.add_argument("--task", default="Isaac-OP3-Teleop-Direct-v0")
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
    parser.add_argument("--use_teacher", action="store_true", help="Play the privileged teacher instead of the deployment student.")
    parser.add_argument(
        "--sample_actions",
        action="store_true",
        help="Sample actions from the privileged teacher instead of using its deterministic mean.",
    )
    parser.add_argument("--print_stats_every", type=int, default=0, help="Print observation and action statistics every N steps.")
    parser.add_argument("--video", action="store_true", help="Record an MP4 clip with gymnasium RecordVideo.")
    parser.add_argument("--video_length", type=int, default=1000)
    parser.add_argument("--video_dir", default=None)
    parser.add_argument("--video_prefix", default="op3_add_playback")
    parser.add_argument(
        "--record_viser",
        default=None,
        help="Optional path for a .viser recording when using the Viser visualizer backend.",
    )
    parser.add_argument("--viser_port", type=int, default=8080)
    parser.add_argument("--viser_share", action="store_true")
    parser.add_argument("--viser_max_worlds", type=int, default=None)
    return parser


def _format_action_stats(action_names: list[str], actions) -> str:
    first_env_actions = actions[0].detach().cpu().tolist()
    return ", ".join(f"{name}={value:+.4f}" for name, value in zip(action_names, first_env_actions, strict=True))


def main() -> None:
    parser = build_arg_parser()
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # type: ignore[no-redef]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if args.sample_actions and not args.use_teacher:
        raise ValueError("--sample_actions is only supported together with --use_teacher.")
    enable_viser = (
        args.record_viser is not None
        or args.viser_share
        or args.viser_port != 8080
        or args.viser_max_worlds is not None
    )
    if enable_viser and getattr(args, "headless", False):
        raise ValueError("Viser playback/recording requires visualizers to be enabled. Omit --headless when using --record_viser.")
    if args.video and hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch

    import op3_teleop_lab.tasks  # noqa: F401
    from op3_teleop_lab.learning.add.config import ADDTrainingConfig
    from op3_teleop_lab.learning.add.trainer import ADDTrainer
    from op3_teleop_lab.tasks.direct.op3_teleop.env_cfg import OP3TeleopEnvCfg, OP3TeleopNewtonEnvCfg

    if args.teleop_mode is None and args.teleop_dataset_path is None:
        default_dataset = Path(__file__).resolve().parents[2] / "data/processed/open/teleop_sparse_pose.npz"
        fallback_dataset = Path(__file__).resolve().parents[2] / "data/processed/open/aist_sparse_pose.npz"
        if default_dataset.exists():
            args.teleop_mode = "dataset"
            args.teleop_dataset_path = str(default_dataset)
        elif fallback_dataset.exists():
            args.teleop_mode = "dataset"
            args.teleop_dataset_path = str(fallback_dataset)

    cfg = OP3TeleopNewtonEnvCfg() if "Newton" in args.task else OP3TeleopEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.teleop_mode is not None:
        cfg.teleop_mode = args.teleop_mode
    if args.teleop_dataset_path is not None:
        cfg.teleop_dataset_path = args.teleop_dataset_path

    if enable_viser:
        try:
            from isaaclab_visualizers.viser import ViserVisualizerCfg
        except ImportError:
            from isaaclab.visualizers.viser import ViserVisualizerCfg  # type: ignore[no-redef]

        cfg.sim.visualizer_cfgs = [
            ViserVisualizerCfg(
                port=args.viser_port,
                share=args.viser_share,
                record_to_viser=args.record_viser,
                max_worlds=args.viser_max_worlds,
            )
        ]

    train_cfg = ADDTrainingConfig.from_yaml(args.config)
    render_mode = "rgb_array" if args.video else None
    env = gym.make(args.task, cfg=cfg, render_mode=render_mode)
    if args.video:
        from gymnasium.wrappers import RecordVideo

        checkpoint_dir = Path(args.checkpoint).resolve().parent
        video_dir = Path(args.video_dir) if args.video_dir is not None else checkpoint_dir / "videos" / "play"
        video_dir.mkdir(parents=True, exist_ok=True)
        env = RecordVideo(
            env,
            video_folder=str(video_dir),
            name_prefix=args.video_prefix,
            step_trigger=lambda step: step == 0,
            video_length=min(args.video_length, args.steps),
            disable_logger=True,
        )
        print(f"Recording playback video to: {video_dir}")

    base_env = env.unwrapped
    obs_dict, extras = env.reset()
    actor_obs = obs_dict["policy"]
    critic_obs = obs_dict.get("critic", actor_obs)
    diff = extras.get("add_diff")
    if diff is None:
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
        out_dir=Path(args.checkpoint).resolve().parent,
        critic_obs_dim=critic_obs.shape[-1],
    )
    trainer.load(args.checkpoint)

    actor_obs = trainer.actor_obs
    action_names = list(getattr(base_env, "action_joint_names", base_env.cfg.profile.joint_names))

    with torch.no_grad():
        for step in range(args.steps):
            if args.use_teacher:
                actions = trainer.teacher_actions(critic_obs, sample=args.sample_actions)
            else:
                actions = trainer.deployment_actions(actor_obs)

            if args.print_stats_every > 0 and step % args.print_stats_every == 0:
                action_abs = actions.abs()
                print(
                    f"[play step {step}] obs_abs_mean={actor_obs.abs().mean().item():.6f} "
                    f"obs_abs_max={actor_obs.abs().max().item():.6f} "
                    f"action_abs_mean={action_abs.mean().item():.6f} "
                    f"action_abs_max={action_abs.max().item():.6f}"
                )
                print(f"[play step {step}] actions_env0: {_format_action_stats(action_names, actions)}")

            obs_dict, _, _, _, _ = env.step(actions)
            actor_obs = obs_dict["policy"]
            critic_obs = obs_dict.get("critic", actor_obs)

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
