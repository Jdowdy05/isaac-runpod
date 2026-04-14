from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.utils.math import quat_apply

from .constants import SEGMENT_INDEX, TRACKED_SEGMENTS
from .env_cfg import OP3TeleopEnvCfg
from .teleop_command import SparsePoseBatch, SparsePoseCommandGenerator


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] *= -1.0
    return result


def quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    w0, x0, y0, z0 = q0.unbind(dim=-1)
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    return torch.stack(
        (
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ),
        dim=-1,
    )


def quat_normalize(quat: torch.Tensor) -> torch.Tensor:
    return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=1.0e-6)


def quaternion_to_tangent_and_normal(quat: torch.Tensor) -> torch.Tensor:
    ref_tangent = torch.zeros_like(quat[..., :3])
    ref_normal = torch.zeros_like(quat[..., :3])
    ref_tangent[..., 0] = 1.0
    ref_normal[..., 2] = 1.0
    tangent = quat_apply(quat, ref_tangent)
    normal = quat_apply(quat, ref_normal)
    return torch.cat((tangent, normal), dim=-1)


class OP3TeleopEnv(DirectRLEnv):
    cfg: OP3TeleopEnvCfg

    def __init__(self, cfg: OP3TeleopEnvCfg, render_mode: str | None = None, **kwargs):
        self.teleop_command: SparsePoseBatch | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        self._joint_ids, _ = self.robot.find_joints(list(self.cfg.profile.joint_names))
        self._joint_ids = [int(joint_id) for joint_id in self._joint_ids]
        self._joint_ids_tensor = torch.tensor(self._joint_ids, dtype=torch.long, device=self.device)
        self._body_ids = self._resolve_body_ids()
        self._root_body_id = self._body_ids["pelvis"]

        self._joint_lower, self._joint_upper = self._gather_joint_limits()
        self._default_joint_pos = self._select_joint_columns(self.robot.data.default_joint_pos).clone()
        self._default_joint_vel = self._select_joint_columns(self.robot.data.default_joint_vel).clone()

        self.actions = torch.zeros(self.num_envs, len(self._joint_ids), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.position_targets = self._default_joint_pos.clone()

        self.command_generator = SparsePoseCommandGenerator(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.cfg.sim.dt * self.cfg.decimation,
            mode=self.cfg.teleop_mode,
            dataset_path=self.cfg.teleop_dataset_path,
        )
        self.teleop_command = self.command_generator.step()

    def _resolve_body_ids(self) -> dict[str, int]:
        body_ids: dict[str, int] = {}
        for segment_name, body_name in self.cfg.profile.segment_to_body_name.items():
            ids, _ = self.robot.find_bodies(body_name)
            if len(ids) != 1:
                raise ValueError(
                    f"Expected exactly one body match for segment '{segment_name}' using name '{body_name}', "
                    f"but found {len(ids)}. Update robot_profile.py to match the final OP3 asset."
                )
            body_ids[segment_name] = int(ids[0])
        return body_ids

    def _setup_scene(self) -> None:
        self.robot = Articulation(self.cfg.robot)
        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self.terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        self.scene.articulations["robot"] = self.robot
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.8, 0.8, 0.8))
        light_cfg.func("/World/Light", light_cfg)

    @staticmethod
    def _scalar_to_float(value) -> float:
        if hasattr(value, "item"):
            return float(value.item())
        return float(value)

    def _as_torch(self, value) -> torch.Tensor:
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

        return torch.as_tensor(value, device=self.device)

    def _gather_joint_limits(self) -> tuple[torch.Tensor, torch.Tensor]:
        limits = self._as_torch(self.robot.data.soft_joint_pos_limits)
        if limits.ndim == 3:
            limits = limits[0]
        if limits.ndim != 2 or limits.shape[-1] != 2:
            raise ValueError(f"Unsupported soft_joint_pos_limits shape: {tuple(limits.shape)}")

        lower = torch.index_select(limits[:, 0], dim=0, index=self._joint_ids_tensor)
        upper = torch.index_select(limits[:, 1], dim=0, index=self._joint_ids_tensor)
        return lower, upper

    def _select_joint_columns(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self._as_torch(tensor)
        return torch.index_select(tensor, dim=-1, index=self._joint_ids_tensor)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()
        unclipped_targets = self._default_joint_pos + self.cfg.action_scale * self.actions
        self.position_targets = torch.clamp(unclipped_targets, self._joint_lower, self._joint_upper)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.position_targets, joint_ids=self._joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.teleop_command = self.command_generator.step()

        joint_pos = self._select_joint_columns(self.robot.data.joint_pos)
        joint_vel = self._select_joint_columns(self.robot.data.joint_vel)
        root_ang_vel = self._as_torch(self.robot.data.root_ang_vel_b)
        projected_gravity = self._as_torch(self.robot.data.projected_gravity_b)
        joint_pos_scaled = self._scale_joint_pos(joint_pos)
        sparse_pose = self.teleop_command.flatten()
        phase_obs = torch.stack((torch.sin(self.teleop_command.phase), torch.cos(self.teleop_command.phase)), dim=-1)

        obs = torch.cat(
            (
                root_ang_vel,
                projected_gravity,
                joint_pos_scaled,
                joint_vel * self.cfg.joint_vel_scale,
                self.prev_actions,
                sparse_pose,
                phase_obs,
            ),
            dim=-1,
        )
        task_reward = getattr(
            self,
            "reward_buf",
            torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        )
        self.extras = {
            "add_diff": self._compute_add_differential(),
            "task_reward": task_reward.clone(),
        }
        return {"policy": obs}

    def _get_rewards(self) -> torch.Tensor:
        root_pos = self._as_torch(self.robot.data.root_pos_w)
        root_quat = quat_normalize(self._as_torch(self.robot.data.root_quat_w))
        root_quat_inv = quat_conjugate(root_quat)

        body_pos_w = self._as_torch(self.robot.data.body_pos_w)
        body_quat_w = quat_normalize(self._as_torch(self.robot.data.body_quat_w))

        pose_pos_reward = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        pose_rot_reward = torch.zeros_like(pose_pos_reward)

        for segment_name in TRACKED_SEGMENTS:
            body_id = self._body_ids[segment_name]
            seg_idx = SEGMENT_INDEX[segment_name]
            pos_valid = self.teleop_command.position_valid[:, seg_idx].float()
            rot_valid = self.teleop_command.rotation_valid[:, seg_idx].float()

            if segment_name == "pelvis":
                pos_error = torch.linalg.norm(body_pos_w[:, body_id] - root_pos, dim=-1)
                current_quat = root_quat
            else:
                current_pos_rel = body_pos_w[:, body_id] - root_pos
                pos_error = torch.linalg.norm(current_pos_rel - self.teleop_command.positions[:, seg_idx], dim=-1)
                current_quat = quat_normalize(quat_mul(root_quat_inv, body_quat_w[:, body_id]))

            pose_pos_reward += pos_valid * torch.exp(-self.cfg.pose_tracking_sigma * pos_error.square())

            target_quat = quat_normalize(self.teleop_command.orientations[:, seg_idx])
            quat_alignment = torch.abs(torch.sum(current_quat * target_quat, dim=-1))
            quat_error = 1.0 - torch.clamp(quat_alignment, 0.0, 1.0)
            pose_rot_reward += rot_valid * torch.exp(-self.cfg.body_orientation_sigma * quat_error.square())

        root_lin_vel_xy = self._as_torch(self.robot.data.root_lin_vel_b)[:, :2]
        cmd_vel_xy = self.teleop_command.target_lin_vel_xy
        locomotion_error = torch.linalg.norm(root_lin_vel_xy - cmd_vel_xy, dim=-1)
        locomotion_reward = torch.exp(-4.0 * locomotion_error.square())

        projected_gravity = self._as_torch(self.robot.data.projected_gravity_b)
        upright_reward = torch.clamp((-projected_gravity[:, 2]), min=0.0, max=1.0)

        action_rate_penalty = torch.sum((self.actions - self.prev_actions).square(), dim=-1)
        joint_pos = self._select_joint_columns(self.robot.data.joint_pos)
        joint_vel = self._select_joint_columns(self.robot.data.joint_vel)
        energy_penalty = torch.sum((self.actions * joint_vel).square(), dim=-1)
        joint_limit_penalty = torch.sum(
            (joint_pos <= self._joint_lower + 1.0e-3)
            | (joint_pos >= self._joint_upper - 1.0e-3),
            dim=-1,
        ).float()

        return (
            self.cfg.alive_reward
            + self.cfg.pose_pos_weight * pose_pos_reward
            + self.cfg.pose_rot_weight * pose_rot_reward
            + self.cfg.locomotion_weight * locomotion_reward
            + self.cfg.upright_weight * upright_reward
            - self.cfg.action_rate_weight * action_rate_penalty
            - self.cfg.energy_weight * energy_penalty
            - self.cfg.joint_limit_weight * joint_limit_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_height = self._as_torch(self.robot.data.root_pos_w)[:, 2]
        upright_cos = -self._as_torch(self.robot.data.projected_gravity_b)[:, 2]
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = (root_height < self.cfg.profile.termination_height) | (
            upright_cos < self.cfg.profile.max_root_tilt_cos
        )
        return fallen, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None) -> None:
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        else:
            env_ids = self._as_torch(env_ids).to(device=self.device, dtype=torch.long)

        self.robot.reset(env_ids)
        super()._reset_idx(env_ids)

        joint_pos = self._default_joint_pos[env_ids].clone()
        joint_vel = self._default_joint_vel[env_ids].clone()
        joint_pos += 0.05 * torch.randn_like(joint_pos)
        joint_pos = torch.clamp(joint_pos, self._joint_lower, self._joint_upper)

        default_root_state = self._as_torch(self.robot.data.default_root_state)[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)
        self.robot.set_joint_position_target(joint_pos, joint_ids=self._joint_ids, env_ids=env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.position_targets[env_ids] = joint_pos
        self.command_generator.reset(env_ids)

    def _scale_joint_pos(self, joint_pos: torch.Tensor) -> torch.Tensor:
        return 2.0 * (joint_pos - self._joint_lower) / torch.clamp(self._joint_upper - self._joint_lower, min=1.0e-6) - 1.0

    def _compute_add_differential(self) -> torch.Tensor:
        root_pos = self._as_torch(self.robot.data.root_pos_w)
        root_quat = quat_normalize(self._as_torch(self.robot.data.root_quat_w))
        root_quat_inv = quat_conjugate(root_quat)
        body_pos_w = self._as_torch(self.robot.data.body_pos_w)
        body_quat_w = quat_normalize(self._as_torch(self.robot.data.body_quat_w))

        pos_diffs = []
        rot_diffs = []
        for segment_name in TRACKED_SEGMENTS:
            body_id = self._body_ids[segment_name]
            seg_idx = SEGMENT_INDEX[segment_name]

            if segment_name == "pelvis":
                current_pos = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
                current_quat = root_quat
            else:
                current_pos = body_pos_w[:, body_id] - root_pos
                current_quat = quat_normalize(quat_mul(root_quat_inv, body_quat_w[:, body_id]))

            target_pos = self.teleop_command.positions[:, seg_idx]
            pos_mask = self.teleop_command.position_valid[:, seg_idx].unsqueeze(-1).float()
            pos_diffs.append((target_pos - current_pos) * pos_mask)

            target_quat = quat_normalize(self.teleop_command.orientations[:, seg_idx])
            target_rot = quaternion_to_tangent_and_normal(target_quat)
            current_rot = quaternion_to_tangent_and_normal(current_quat)
            rot_mask = self.teleop_command.rotation_valid[:, seg_idx].unsqueeze(-1).float()
            rot_diffs.append((target_rot - current_rot) * rot_mask)

        vel_diff = self.teleop_command.target_lin_vel_xy - self._as_torch(self.robot.data.root_lin_vel_b)[:, :2]
        return torch.cat(
            (
                torch.cat(pos_diffs, dim=-1),
                torch.cat(rot_diffs, dim=-1),
                vel_diff,
            ),
            dim=-1,
        )
