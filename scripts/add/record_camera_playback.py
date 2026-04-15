#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Record an OP3 ADD checkpoint with a headless camera sensor.")
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=50)
    parser.add_argument("--camera_distance", type=float, default=2.3)
    parser.add_argument("--camera_side_offset", type=float, default=-1.6)
    parser.add_argument("--camera_height_offset", type=float, default=1.0)
    parser.add_argument("--camera_target_height", type=float, default=0.35)
    parser.add_argument("--output", default=None)
    return parser


def _default_dataset_path() -> tuple[str | None, str | None]:
    root = Path(__file__).resolve().parents[2]
    merged = root / "data/processed/open/teleop_sparse_pose.npz"
    aist = root / "data/processed/open/aist_sparse_pose.npz"
    if merged.exists():
        return "dataset", str(merged)
    if aist.exists():
        return "dataset", str(aist)
    return None, None


def _frame_to_uint8(frame):
    import numpy as np

    if frame.dtype == np.uint8:
        return frame
    frame = frame[..., :3]
    frame = np.clip(frame, 0, 255)
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return frame.astype(np.uint8)


def main() -> None:
    parser = build_arg_parser()
    try:
        from isaaclab.app import AppLauncher
    except ImportError:
        from omni.isaac.lab.app import AppLauncher  # type: ignore[no-redef]

    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = True
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from PIL import Image

    import isaaclab.sim as sim_utils
    from isaaclab.sensors.camera import Camera, CameraCfg
    from op3_teleop_lab.learning.add.config import ADDTrainingConfig
    from op3_teleop_lab.learning.add.trainer import ADDTrainer
    from op3_teleop_lab.tasks.direct.op3_teleop.env_cfg import OP3TeleopEnvCfg, OP3TeleopNewtonEnvCfg
    import op3_teleop_lab.tasks  # noqa: F401

    if args.teleop_mode is None and args.teleop_dataset_path is None:
        args.teleop_mode, args.teleop_dataset_path = _default_dataset_path()

    cfg = OP3TeleopNewtonEnvCfg() if "Newton" in args.task else OP3TeleopEnvCfg()
    cfg.scene.num_envs = args.num_envs
    if args.teleop_mode is not None:
        cfg.teleop_mode = args.teleop_mode
    if args.teleop_dataset_path is not None:
        cfg.teleop_dataset_path = args.teleop_dataset_path

    train_cfg = ADDTrainingConfig.from_yaml(args.config)
    env = gym.make(args.task, cfg=cfg)
    base_env = env.unwrapped

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

    output_path = Path(args.output) if args.output is not None else Path(args.checkpoint).resolve().parent / "videos" / "camera_playback.mp4"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise FileNotFoundError("ffmpeg is required to encode the playback video.")

    frames_dir = output_path.with_name(f"{output_path.stem}_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    actor_obs = trainer.actor_obs
    with torch.no_grad():
        for step in range(args.steps):
            root_pos = base_env._as_torch(base_env.robot.data.root_pos_w)[0]
            camera_position = root_pos + torch.tensor(
                [args.camera_distance, args.camera_side_offset, args.camera_height_offset],
                device=actor_obs.device,
            )
            camera_target = root_pos + torch.tensor(
                [0.0, 0.0, args.camera_target_height],
                device=actor_obs.device,
            )
            camera.set_world_poses_from_view(camera_position.unsqueeze(0), camera_target.unsqueeze(0))

            actions = trainer.policy.deterministic(actor_obs)
            obs_dict, _, _, _, _ = env.step(actions)
            actor_obs = obs_dict["policy"]

            camera.update(dt=base_env.physics_dt)
            frame = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
            Image.fromarray(_frame_to_uint8(frame)).save(frames_dir / f"frame_{step:05d}.png")

    ffmpeg_cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(args.fps),
        "-i",
        str(frames_dir / "frame_%05d.png"),
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    subprocess.run(ffmpeg_cmd, check=True)

    print(f"Saved camera playback video to: {output_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
