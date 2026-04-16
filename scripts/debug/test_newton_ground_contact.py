#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
from pathlib import Path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Standalone Newton diagnostic for OP3 ground contact and root height."
    )
    parser.add_argument("--steps", type=int, default=2000, help="Maximum number of physics steps to simulate.")
    parser.add_argument("--print_every", type=int, default=10, help="Print diagnostics every N simulation steps.")
    parser.add_argument("--z_fail", type=float, default=-5.0, help="Abort if root height drops below this value.")
    parser.add_argument(
        "--hold-default-pose",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Continuously apply the default joint position target.",
    )
    return parser


def _summarize_ground_prim(prim) -> dict[str, object]:
    from pxr import UsdPhysics

    if not prim.IsValid():
        return {"valid": False, "path": str(prim.GetPath())}

    stack = [prim]
    collision_prims: list[str] = []
    while stack:
        current = stack.pop()
        if current.HasAPI(UsdPhysics.CollisionAPI):
            collision_prims.append(current.GetPath().pathString)
        stack.extend(list(current.GetChildren()))

    return {
        "valid": True,
        "path": prim.GetPath().pathString,
        "type_name": prim.GetTypeName(),
        "children": [child.GetPath().pathString for child in prim.GetChildren()],
        "collision_prims": collision_prims,
    }


def _resolve_body_id(robot, body_name: str) -> int:
    body_ids, body_names = robot.find_bodies(body_name)
    if len(body_ids) != 1:
        raise ValueError(f"Expected exactly one body match for '{body_name}', got {list(body_names)}")
    return int(body_ids[0])


def _as_torch(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value
    try:
        import warp as wp  # type: ignore

        try:
            return wp.to_torch(value)
        except Exception:
            if hasattr(value, "contiguous"):
                return wp.to_torch(value.contiguous())
            raise
    except ImportError:
        pass
    return torch.as_tensor(value, device=device)


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

    import torch
    import omni.usd
    import isaaclab.sim as sim_utils
    from isaaclab.assets import Articulation

    from op3_teleop_lab.assets.op3 import resolve_op3_cfg
    from op3_teleop_lab.tasks.direct.op3_teleop.env_cfg import OP3TeleopNewtonEnvCfg
    from op3_teleop_lab.utils.physics import build_sim_cfg

    cfg = OP3TeleopNewtonEnvCfg()
    cfg.scene.num_envs = 1
    cfg.terrain.num_envs = 1
    cfg.terrain.env_spacing = cfg.scene.env_spacing
    cfg.terrain.debug_vis = False

    sim_cfg = build_sim_cfg("newton", dt=cfg.sim.dt, render_interval=1)
    sim = sim_utils.SimulationContext(sim_cfg)

    if not getattr(args, "headless", False):
        sim.set_camera_view([2.5, -2.0, 1.4], [0.0, 0.0, 0.35])

    terrain = cfg.terrain.class_type(cfg.terrain)
    robot_cfg = resolve_op3_cfg().replace(prim_path="/World/Robot")
    robot = Articulation(robot_cfg)

    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
    light_cfg.func("/World/Light", light_cfg)

    sim.reset()
    robot.reset()

    stage = omni.usd.get_context().get_stage()
    ground_summary = _summarize_ground_prim(stage.GetPrimAtPath(cfg.terrain.prim_path))
    print(json.dumps({"ground_summary": ground_summary}, sort_keys=True))

    default_root_state = _as_torch(robot.data.default_root_state, robot.device).clone()
    if default_root_state.ndim == 1:
        default_root_state = default_root_state.unsqueeze(0)
    default_joint_pos = _as_torch(robot.data.default_joint_pos, robot.device).clone()
    default_joint_vel = _as_torch(robot.data.default_joint_vel, robot.device).clone()
    if default_joint_pos.ndim == 1:
        default_joint_pos = default_joint_pos.unsqueeze(0)
    if default_joint_vel.ndim == 1:
        default_joint_vel = default_joint_vel.unsqueeze(0)

    robot.write_root_pose_to_sim(default_root_state[:, :7])
    robot.write_root_velocity_to_sim(default_root_state[:, 7:])
    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
    robot.reset()

    left_foot_id = _resolve_body_id(robot, "l_ank_roll_link")
    right_foot_id = _resolve_body_id(robot, "r_ank_roll_link")
    sim_dt = cfg.sim.dt

    for step in range(args.steps):
        if args.hold_default_pose:
            robot.set_joint_position_target(default_joint_pos)
        robot.write_data_to_sim()
        sim.step()
        robot.update(sim_dt)

        root_pos_w = _as_torch(robot.data.root_pos_w, robot.device)[0]
        root_quat_w = _as_torch(robot.data.root_quat_w, robot.device)[0]
        body_pos_w = _as_torch(robot.data.body_pos_w, robot.device)
        left_foot_z = float(body_pos_w[0, left_foot_id, 2].item())
        right_foot_z = float(body_pos_w[0, right_foot_id, 2].item())
        root_z = float(root_pos_w[2].item())

        if step % args.print_every == 0 or root_z < args.z_fail:
            print(
                json.dumps(
                    {
                        "step": step,
                        "root_pos_w": [float(v) for v in root_pos_w.tolist()],
                        "root_quat_w": [float(v) for v in root_quat_w.tolist()],
                        "root_z": root_z,
                        "left_foot_z": left_foot_z,
                        "right_foot_z": right_foot_z,
                    },
                    sort_keys=True,
                )
            )

        if torch.isnan(root_pos_w).any():
            print(json.dumps({"status": "nan_detected", "step": step}, sort_keys=True))
            break
        if root_z < args.z_fail:
            print(json.dumps({"status": "fell_through_ground", "step": step, "root_z": root_z}, sort_keys=True))
            break
    else:
        print(json.dumps({"status": "completed", "steps": args.steps}, sort_keys=True))

    simulation_app.close()


if __name__ == "__main__":
    main()
