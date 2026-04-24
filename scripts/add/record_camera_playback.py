#!/usr/bin/env python3

from __future__ import annotations

import argparse
import shutil
import subprocess
from pathlib import Path

_STATE_CONNECTIONS = (
    ("pelvis", "head"),
    ("pelvis", "left_hand"),
    ("pelvis", "right_hand"),
    ("pelvis", "left_knee"),
    ("left_knee", "left_foot"),
    ("pelvis", "right_knee"),
    ("right_knee", "right_foot"),
)


def build_arg_parser():
    parser = argparse.ArgumentParser(description="Record an OP3 ADD checkpoint with a headless camera sensor.")
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
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--use_teacher", action="store_true", help="Record the privileged teacher instead of the deployment student.")
    parser.add_argument(
        "--sample_actions",
        action="store_true",
        help="Sample actions from the privileged teacher instead of using its deterministic mean.",
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


def _format_action_stats(action_names: list[str], actions) -> str:
    first_env_actions = actions[0].detach().cpu().tolist()
    return ", ".join(f"{name}={value:+.4f}" for name, value in zip(action_names, first_env_actions, strict=True))


def _format_joint_target_debug(base_env, action_names: list[str], actions) -> str:
    import torch

    current_joint_pos = base_env._select_joint_columns(base_env.robot.data.joint_pos)[0].detach().cpu()
    default_joint_pos = base_env._default_joint_pos[0].detach().cpu()
    joint_lower = base_env._joint_lower.detach().cpu()
    joint_upper = base_env._joint_upper.detach().cpu()
    clipped_actions = torch.clamp(
        actions[0].detach().cpu(),
        -float(getattr(base_env.cfg, "action_clip", 1.0)),
        float(getattr(base_env.cfg, "action_clip", 1.0)),
    )
    target_joint_pos = torch.clamp(default_joint_pos + float(base_env.cfg.action_scale) * clipped_actions, joint_lower, joint_upper)
    target_error = (target_joint_pos - current_joint_pos).abs()
    joint_from_default = (current_joint_pos - default_joint_pos).abs()
    top_k = min(4, target_error.numel())
    top_indices = torch.topk(target_error, k=top_k).indices.tolist()
    top_joint_errors = ", ".join(
        (
            f"{action_names[idx]} cur={current_joint_pos[idx]:+.3f} "
            f"tgt={target_joint_pos[idx]:+.3f} err={target_error[idx]:.3f}"
        )
        for idx in top_indices
    )
    return (
        f"target_abs_mean={target_joint_pos.abs().mean().item():.6f} "
        f"target_error_abs_mean={target_error.mean().item():.6f} "
        f"target_error_abs_max={target_error.max().item():.6f} "
        f"joint_from_default_abs_mean={joint_from_default.mean().item():.6f} | "
        f"{top_joint_errors}"
    )


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


def _encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is not None:
        ffmpeg_cmd = [
            ffmpeg,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            str(frames_dir / "frame_%05d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ]
        subprocess.run(ffmpeg_cmd, check=True)
        return

    try:
        import imageio.v2 as imageio
    except ImportError as exc:
        raise FileNotFoundError("ffmpeg or imageio is required to encode the playback video.") from exc

    print("ffmpeg was not found; using imageio/imageio-ffmpeg to encode the playback video.")
    frame_paths = sorted(frames_dir.glob("frame_*.png"))
    if not frame_paths:
        raise FileNotFoundError(f"No playback frames found in {frames_dir}.")
    with imageio.get_writer(str(output_path), fps=fps, codec="libx264", macro_block_size=1) as writer:
        for frame_path in frame_paths:
            writer.append_data(imageio.imread(frame_path))


def _draw_state_frame(
    *,
    base_env,
    tracked_segments: tuple[str, ...],
    segment_index: dict[str, int],
    actions,
    step: int,
    width: int,
    height: int,
):
    from PIL import Image, ImageDraw
    import torch

    image = Image.new("RGB", (width, height), (247, 248, 245))
    draw = ImageDraw.Draw(image)

    root_pos = base_env._as_torch(base_env.robot.data.root_pos_w)[0].detach().cpu()
    body_pos_w = base_env._as_torch(base_env.robot.data.body_pos_w)[0].detach().cpu()
    command_pos = base_env.teleop_command.positions[0].detach().cpu()

    actual = {}
    target = {}
    for segment_name in tracked_segments:
        seg_idx = segment_index[segment_name]
        if segment_name == "pelvis":
            actual[segment_name] = torch.zeros(3)
        else:
            body_id = base_env._body_ids[segment_name]
            actual[segment_name] = body_pos_w[body_id] - root_pos
        target[segment_name] = command_pos[seg_idx]

    root_height = float(root_pos[2].item())
    action_abs_mean = float(actions.abs().mean().item())
    action_abs_max = float(actions.abs().max().item())

    header = (
        f"checkpoint playback | step {step:04d} | root_z={root_height:.3f} m | "
        f"action_abs_mean={action_abs_mean:.4f} | action_abs_max={action_abs_max:.4f}"
    )
    draw.text((24, 16), header, fill=(24, 28, 32))

    panels = (
        ("side view: x-z", (24, 56, width // 2 - 16, height - 30), (0, 2)),
        ("top view: x-y", (width // 2 + 16, 56, width - 24, height - 30), (0, 1)),
    )
    colors = {
        "actual": (37, 97, 180),
        "target": (218, 121, 38),
        "grid": (218, 221, 216),
        "axis": (90, 95, 90),
        "floor": (58, 66, 60),
    }

    def project(point, panel, axes, scale):
        left, top, right, bottom = panel
        cx = (left + right) * 0.5
        cy = (top + bottom) * 0.54
        x = cx + float(point[axes[0]].item()) * scale
        y = cy - float(point[axes[1]].item()) * scale
        return (int(round(x)), int(round(y)))

    def draw_pose(points, panel, axes, color, width_px):
        scale = min((panel[2] - panel[0]) / 1.1, (panel[3] - panel[1]) / 0.9)
        for a, b in _STATE_CONNECTIONS:
            if a in points and b in points:
                draw.line((project(points[a], panel, axes, scale), project(points[b], panel, axes, scale)), fill=color, width=width_px)
        for name, point in points.items():
            px, py = project(point, panel, axes, scale)
            radius = 6 if name == "pelvis" else 4
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color)

    for label, panel, axes in panels:
        left, top, right, bottom = panel
        draw.rounded_rectangle(panel, radius=14, fill=(255, 255, 252), outline=(208, 211, 204), width=1)
        draw.text((left + 14, top + 12), label, fill=(24, 28, 32))
        scale = min((right - left) / 1.1, (bottom - top) / 0.9)
        center_x = int(round((left + right) * 0.5))
        center_y = int(round((top + bottom) * 0.54))
        for offset in range(-4, 5):
            x = int(round(center_x + offset * 0.1 * scale))
            y = int(round(center_y + offset * 0.1 * scale))
            draw.line((x, top + 42, x, bottom - 12), fill=colors["grid"], width=1)
            draw.line((left + 12, y, right - 12, y), fill=colors["grid"], width=1)
        draw.line((left + 12, center_y, right - 12, center_y), fill=colors["axis"], width=1)
        draw.line((center_x, top + 42, center_x, bottom - 12), fill=colors["axis"], width=1)
        if axes == (0, 2):
            floor_y = int(round(center_y + root_height * scale))
            draw.line((left + 12, floor_y, right - 12, floor_y), fill=colors["floor"], width=2)
            draw.text((left + 14, min(floor_y + 4, bottom - 26)), "floor z=0", fill=colors["floor"])

        draw_pose(target, panel, axes, colors["target"], 3)
        draw_pose(actual, panel, axes, colors["actual"], 4)

    legend_y = height - 22
    draw.line((24, legend_y, 60, legend_y), fill=colors["actual"], width=4)
    draw.text((68, legend_y - 8), "actual OP3 sparse bodies", fill=(24, 28, 32))
    draw.line((260, legend_y, 296, legend_y), fill=colors["target"], width=3)
    draw.text((304, legend_y - 8), "target sparse command", fill=(24, 28, 32))
    return image


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
    if hasattr(args, "enable_cameras"):
        args.enable_cameras = not args.state_video
    app_launcher = AppLauncher(args)
    simulation_app = app_launcher.app

    import gymnasium as gym
    import torch
    from PIL import Image

    from op3_teleop_lab.learning.add.config import ADDTrainingConfig
    from op3_teleop_lab.learning.add.trainer import ADDTrainer
    from op3_teleop_lab.tasks.direct.op3_teleop.constants import SEGMENT_INDEX, TRACKED_SEGMENTS
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

    frames_dir = output_path.with_name(f"{output_path.stem}_frames")
    if frames_dir.exists():
        shutil.rmtree(frames_dir)
    frames_dir.mkdir(parents=True, exist_ok=True)

    actor_obs = trainer.actor_obs
    action_names = list(getattr(base_env, "action_joint_names", base_env.cfg.profile.joint_names))
    use_state_video = args.state_video
    with torch.no_grad():
        for step in range(args.steps):
            root_pos = base_env._as_torch(base_env.robot.data.root_pos_w)[0]
            if not use_state_video and camera is not None:
                camera_position = root_pos + torch.tensor(
                    [args.camera_distance, args.camera_side_offset, args.camera_height_offset],
                    device=actor_obs.device,
                )
                camera_target = root_pos + torch.tensor(
                    [0.0, 0.0, args.camera_target_height],
                    device=actor_obs.device,
                )
                camera.set_world_poses_from_view(camera_position.unsqueeze(0), camera_target.unsqueeze(0))

            if args.use_teacher:
                actions = trainer.teacher_actions(critic_obs, sample=args.sample_actions)
            else:
                actions = trainer.deployment_actions(actor_obs)

            if args.print_stats_every > 0 and step % args.print_stats_every == 0:
                action_abs = actions.abs()
                print(
                    f"[record step {step}] obs_abs_mean={actor_obs.abs().mean().item():.6f} "
                    f"obs_abs_max={actor_obs.abs().max().item():.6f} "
                    f"action_abs_mean={action_abs.mean().item():.6f} "
                    f"action_abs_max={action_abs.max().item():.6f}"
                )
                print(f"[record step {step}] actions_env0: {_format_action_stats(action_names, actions)}")
                print(f"[record step {step}] target_debug_env0: {_format_joint_target_debug(base_env, action_names, actions)}")
            obs_dict, _, _, _, _ = env.step(actions)
            actor_obs = obs_dict["policy"]
            critic_obs = obs_dict.get("critic", actor_obs)

            if use_state_video or camera is None:
                frame_image = _draw_state_frame(
                    base_env=base_env,
                    tracked_segments=TRACKED_SEGMENTS,
                    segment_index=SEGMENT_INDEX,
                    actions=actions,
                    step=step,
                    width=args.width,
                    height=args.height,
                )
            else:
                try:
                    camera.update(dt=base_env.physics_dt)
                    frame = camera.data.output["rgb"][0].detach().cpu().numpy()[..., :3]
                    frame_uint8 = _frame_to_uint8(frame)
                    if frame_uint8.ndim != 3 or frame_uint8.shape[0] == 0 or frame_uint8.shape[1] == 0:
                        raise ValueError(f"Invalid camera frame shape: {tuple(frame_uint8.shape)}")
                    frame_image = Image.fromarray(frame_uint8)
                except Exception as exc:
                    print(f"Camera capture failed at step {step}: {exc}. Falling back to renderer-free state video.")
                    use_state_video = True
                    frame_image = _draw_state_frame(
                        base_env=base_env,
                        tracked_segments=TRACKED_SEGMENTS,
                        segment_index=SEGMENT_INDEX,
                        actions=actions,
                        step=step,
                        width=args.width,
                        height=args.height,
                    )
            frame_image.save(frames_dir / f"frame_{step:05d}.png")

    _encode_video(frames_dir, output_path, args.fps)

    print(f"Saved camera playback video to: {output_path}")

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
