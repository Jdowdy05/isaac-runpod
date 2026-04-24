from __future__ import annotations

import shutil
import subprocess
from importlib import metadata
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


def default_dataset_path() -> tuple[str | None, str | None]:
    root = Path(__file__).resolve().parents[2]
    merged = root / "data/processed/open/teleop_sparse_pose.npz"
    aist = root / "data/processed/open/aist_sparse_pose.npz"
    if merged.exists():
        return "dataset", str(merged)
    if aist.exists():
        return "dataset", str(aist)
    return None, None


def format_action_stats(action_names: list[str], actions) -> str:
    first_env_actions = actions[0].detach().cpu().tolist()
    return ", ".join(f"{name}={value:+.4f}" for name, value in zip(action_names, first_env_actions, strict=True))


def format_joint_target_debug(base_env, action_names: list[str], actions) -> str:
    import torch

    current_joint_pos = base_env._select_joint_columns(base_env.robot.data.joint_pos)[0].detach().cpu()
    default_joint_pos = base_env._default_joint_pos[0].detach().cpu()
    clipped_actions = torch.clamp(
        actions.detach(),
        -float(getattr(base_env.cfg, "action_clip", 100.0)),
        float(getattr(base_env.cfg, "action_clip", 100.0)),
    )
    target_joint_pos = base_env._actions_to_position_targets(clipped_actions)[0].detach().cpu()
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


def policy_obs_tensor(obs):
    if "policy" in obs.keys():
        return obs["policy"]
    first_key = next(iter(obs.keys()))
    return obs[first_key]


def frame_to_uint8(frame):
    import numpy as np

    if frame.dtype == np.uint8:
        return frame
    frame = frame[..., :3]
    frame = np.clip(frame, 0, 255)
    if frame.max() <= 1.0:
        frame = frame * 255.0
    return frame.astype(np.uint8)


def encode_video(frames_dir: Path, output_path: Path, fps: int) -> None:
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


def draw_state_frame(
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
    from op3_teleop_lab.tasks.direct.op3_teleop.env import quat_apply, quat_conjugate, quat_normalize

    image = Image.new("RGB", (width, height), (247, 248, 245))
    draw = ImageDraw.Draw(image)

    root_pos_t = base_env._as_torch(base_env.robot.data.root_pos_w)[0]
    root_quat_inv = quat_conjugate(quat_normalize(base_env._as_torch(base_env.robot.data.root_quat_w)[0]))
    body_pos_w_t = base_env._as_torch(base_env.robot.data.body_pos_w)[0]
    root_pos = root_pos_t.detach().cpu()
    command_pos = base_env.teleop_command.positions[0].detach().cpu()

    actual = {}
    target = {}
    for segment_name in tracked_segments:
        seg_idx = segment_index[segment_name]
        if segment_name == "pelvis":
            actual[segment_name] = torch.zeros(3)
        else:
            body_id = base_env._body_ids[segment_name]
            actual[segment_name] = quat_apply(root_quat_inv, body_pos_w_t[body_id] - root_pos_t).detach().cpu()
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
                draw.line(
                    (project(points[a], panel, axes, scale), project(points[b], panel, axes, scale)),
                    fill=color,
                    width=width_px,
                )
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


def create_rsl_runner(
    *,
    env,
    runner_kind: str,
    add_config_path: str | None,
    diff_dim: int | None,
    checkpoint_dir: Path,
    device: str | None = None,
):
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper, handle_deprecated_rsl_rl_cfg
    from op3_teleop_lab.learning.add.config import ADDTrainingConfig
    from op3_teleop_lab.learning.rsl_add import RslAddOnPolicyRunner
    from op3_teleop_lab.tasks.direct.op3_teleop.agents.rsl_rl_ppo_cfg import OP3TeleopPPORunnerCfg
    from rsl_rl.runners import OnPolicyRunner

    runner_cfg_obj = OP3TeleopPPORunnerCfg()
    if device is not None:
        runner_cfg_obj.device = device
    runner_cfg_obj = handle_deprecated_rsl_rl_cfg(runner_cfg_obj, metadata.version("rsl-rl-lib"))
    wrapped_env = RslRlVecEnvWrapper(env, clip_actions=runner_cfg_obj.clip_actions)
    runner_cfg = runner_cfg_obj.to_dict()

    if runner_kind == "add":
        if add_config_path is None:
            raise ValueError("RSL-ADD playback requires --add_config.")
        if diff_dim is None:
            raise ValueError("RSL-ADD playback requires a valid add_diff dimension.")
        add_cfg = ADDTrainingConfig.from_yaml(add_config_path)
        runner = RslAddOnPolicyRunner(
            wrapped_env,
            runner_cfg,
            add_cfg=add_cfg,
            diff_dim=diff_dim,
            log_dir=str(checkpoint_dir),
            device=runner_cfg_obj.device,
        )
    elif runner_kind == "ppo":
        runner = OnPolicyRunner(wrapped_env, runner_cfg, log_dir=str(checkpoint_dir), device=runner_cfg_obj.device)
    else:
        raise ValueError(f"Unsupported runner kind: {runner_kind}")

    return wrapped_env, runner, runner_cfg_obj.device
