#!/usr/bin/env python3

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

from common import (
    create_rsl_runner,
    default_dataset_path,
    draw_state_frame,
    encode_video,
    format_action_stats,
    format_joint_target_debug,
    frame_to_uint8,
    policy_obs_tensor,
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Record a sparse humanoid RSL or RSL-ADD checkpoint.")
    parser.add_argument("--task", default="Isaac-OP3-Teleop-Direct-v0")
    parser.add_argument("--runner", choices=("ppo", "add"), default="ppo")
    parser.add_argument("--num_envs", type=int, default=1)
    parser.add_argument("--teleop_mode", choices=("synthetic", "dataset"), default=None)
    parser.add_argument("--teleop_dataset_path", default=None)
    parser.add_argument(
        "--add_config",
        default=None,
    )
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument(
        "--sample_actions",
        action="store_true",
        help="Sample from the stochastic policy distribution instead of taking deterministic means.",
    )
    parser.add_argument("--print_stats_every", type=int, default=0, help="Print observation and action statistics every N steps.")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--camera_distance", type=float, default=2.3)
    parser.add_argument("--camera_side_offset", type=float, default=-1.6)
    parser.add_argument("--camera_height_offset", type=float, default=1.0)
    parser.add_argument("--camera_target_height", type=float, default=0.35)
    parser.add_argument(
        "--state_video",
        action="store_true",
        help="Record a renderer-free 2-D state visualization instead of an Isaac camera render.",
    )
    parser.add_argument("--output", default=None)
    return parser


def main() -> None:
    parser = build_arg_parser()
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # type: ignore[no-redef]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = not args.state_video
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from PIL import Image

    import op3_teleop_lab.tasks  # noqa: F401
    from op3_teleop_lab.tasks.direct.humanoid_teleop.constants import SEGMENT_INDEX, TRACKED_SEGMENTS
    from op3_teleop_lab.tasks.direct.humanoid_teleop.env import quat_apply
    from op3_teleop_lab.tasks.task_registry import make_env_cfg_for_task

    if args.teleop_mode is None and args.teleop_dataset_path is None:
        args.teleop_mode, args.teleop_dataset_path = default_dataset_path()

    cfg = make_env_cfg_for_task(args.task)
    cfg.scene.num_envs = args.num_envs
    if args.teleop_mode is not None:
        cfg.teleop_mode = args.teleop_mode
    if args.teleop_dataset_path is not None:
        cfg.teleop_dataset_path = args.teleop_dataset_path
    if args.runner == "add":
        cfg.add_diff_reward_weight = 0.0

    env = gym.make(args.task, cfg=cfg)
    base_env = env.unwrapped

    camera = None
    if not args.state_video:
        import isaaclab.sim as sim_utils
        from isaaclab.sensors.camera import Camera, CameraCfg

        camera_cfg = CameraCfg(
            prim_path="/World/PlaybackCamera",
            update_period=0,
            height=args.height,
            width=args.width,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=400.0,
                horizontal_aperture=20.955,
                clipping_range=(0.01, 1.0e5),
            ),
        )
        camera = Camera(cfg=camera_cfg)
        if not camera.is_initialized:
            camera._initialize_impl()
            camera._is_initialized = True
        camera.reset()

    obs_dict, extras = env.reset()
    actor_obs = obs_dict["policy"]
    diff = extras.get("add_diff")
    if args.runner == "add" and diff is None:
        zero_action = torch.zeros((base_env.num_envs, env.action_space.shape[-1]), device=actor_obs.device)
        obs_dict, _, _, _, extras = env.step(zero_action)
        diff = extras["add_diff"]

    wrapped_env, runner, runner_device = create_rsl_runner(
        env=env,
        task_name=args.task,
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
    output_path = Path(args.output) if args.output is not None else Path(args.checkpoint).with_suffix(".mp4")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="op3_rsl_playback_") as tmp_dir:
        frames_dir = Path(tmp_dir)
        with torch.no_grad():
            for step in range(args.steps):
                actions = policy(obs, stochastic_output=args.sample_actions)

                if args.print_stats_every > 0 and step % args.print_stats_every == 0:
                    obs_tensor = policy_obs_tensor(obs)
                    action_abs = actions.abs()
                    print(
                        f"[record step {step}] obs_abs_mean={obs_tensor.abs().mean().item():.6f} "
                        f"obs_abs_max={obs_tensor.abs().max().item():.6f} "
                        f"action_abs_mean={action_abs.mean().item():.6f} "
                        f"action_abs_max={action_abs.max().item():.6f}"
                    )
                    print(f"[record step {step}] actions_env0: {format_action_stats(action_names, actions)}")
                    print(
                        f"[record step {step}] target_debug_env0: "
                        f"{format_joint_target_debug(base_env, action_names, actions)}"
                    )

                obs, _, _, _ = wrapped_env.step(actions)

                if args.state_video:
                    frame = draw_state_frame(
                        base_env=base_env,
                        tracked_segments=TRACKED_SEGMENTS,
                        segment_index=SEGMENT_INDEX,
                        actions=actions,
                        step=step,
                        width=args.width,
                        height=args.height,
                    )
                else:
                    robot_root_pos = base_env._as_torch(base_env.robot.data.root_pos_w)[0]
                    robot_root_quat = base_env._as_torch(base_env.robot.data.root_quat_w)[0]
                    forward = base_env._as_torch(base_env.robot.data.projected_gravity_b)[0].new_tensor((1.0, 0.0, 0.0))
                    right = base_env._as_torch(base_env.robot.data.projected_gravity_b)[0].new_tensor((0.0, 1.0, 0.0))

                    camera_offset = (
                        -args.camera_distance * quat_apply(robot_root_quat.unsqueeze(0), forward.unsqueeze(0))[0]
                        + args.camera_side_offset * quat_apply(robot_root_quat.unsqueeze(0), right.unsqueeze(0))[0]
                    )
                    camera_eye = robot_root_pos + camera_offset
                    camera_eye[2] = robot_root_pos[2] + args.camera_height_offset
                    camera_target = robot_root_pos.clone()
                    camera_target[2] = args.camera_target_height
                    camera.set_world_poses_from_view(camera_eye.unsqueeze(0), camera_target.unsqueeze(0))
                    camera.update(dt=0.0)
                    rgb = camera.data.output["rgb"][0].detach().cpu().numpy()
                    frame = Image.fromarray(frame_to_uint8(rgb))

                frame.save(frames_dir / f"frame_{step:05d}.png")

        encode_video(frames_dir, output_path, fps=args.fps)

    print(f"Saved playback video to: {output_path}")
    wrapped_env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
