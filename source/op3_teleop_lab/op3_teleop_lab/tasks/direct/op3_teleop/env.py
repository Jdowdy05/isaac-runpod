from __future__ import annotations

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv

from .constants import SEGMENT_INDEX, SPARSE_POSE_DIM, TRACKED_SEGMENTS
from .env_cfg import OP3TeleopEnvCfg
from .robot_profile import get_action_joint_names
from .teleop_command import SparsePoseBatch, SparsePoseCommandGenerator, quat_from_euler_xyz


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., :3] *= -1.0
    return result


def quat_mul(q0: torch.Tensor, q1: torch.Tensor) -> torch.Tensor:
    x0, y0, z0, w0 = q0.unbind(dim=-1)
    x1, y1, z1, w1 = q1.unbind(dim=-1)
    return torch.stack(
        (
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
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


def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    t = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * t + torch.cross(quat_xyz, t, dim=-1)


class OP3TeleopEnv(DirectRLEnv):
    cfg: OP3TeleopEnvCfg

    def __init__(self, cfg: OP3TeleopEnvCfg, render_mode: str | None = None, **kwargs):
        self.teleop_command: SparsePoseBatch | None = None
        super().__init__(cfg, render_mode=render_mode, **kwargs)

        joint_ids, joint_names = self.robot.find_joints(list(self.cfg.profile.joint_names))
        resolved_joint_map = {str(name): int(joint_id) for joint_id, name in zip(joint_ids, joint_names, strict=True)}
        missing_joint_names = [name for name in self.cfg.profile.joint_names if name not in resolved_joint_map]
        if missing_joint_names:
            raise ValueError(
                "Could not resolve all OP3 joints from the final asset. "
                f"Missing joints: {missing_joint_names}. Resolved joints: {tuple(resolved_joint_map)}"
            )
        self._joint_names = list(get_action_joint_names(self.cfg.profile))
        self.action_joint_names = tuple(self._joint_names)
        self._fixed_joint_names = [name for name in self.cfg.profile.joint_names if name not in self._joint_names]
        self._joint_ids = [resolved_joint_map[name] for name in self._joint_names]
        self._joint_ids_tensor = torch.tensor(self._joint_ids, dtype=torch.long, device=self.device)
        self._fixed_joint_ids = [resolved_joint_map[name] for name in self._fixed_joint_names]
        self._fixed_joint_ids_tensor = torch.tensor(self._fixed_joint_ids, dtype=torch.long, device=self.device)
        self._body_ids = self._resolve_body_ids()
        self._root_body_id = self._body_ids["pelvis"]

        self._joint_lower, self._joint_upper = self._gather_joint_limits()
        self._default_joint_pos = self._select_joint_columns(self.robot.data.default_joint_pos).clone()
        self._default_joint_vel = self._select_joint_columns(self.robot.data.default_joint_vel).clone()
        self._default_fixed_joint_pos = self._select_fixed_joint_columns(self.robot.data.default_joint_pos).clone()
        self._default_fixed_joint_vel = self._select_fixed_joint_columns(self.robot.data.default_joint_vel).clone()
        self._default_joint_stiffness = self._maybe_select_joint_columns("default_joint_stiffness")
        self._default_joint_damping = self._maybe_select_joint_columns("default_joint_damping")
        self._default_root_state = self._resolve_default_root_state()
        self._default_root_height = float(self._default_root_state[0, 2].item())
        self._contact_segments = (
            "left_hand",
            "right_hand",
            "left_knee",
            "right_knee",
            "left_foot",
            "right_foot",
        )
        self._contact_body_ids = torch.tensor(
            [self._body_ids[name] for name in self._contact_segments],
            dtype=torch.long,
            device=self.device,
        )
        self._contact_height_thresholds = torch.tensor(
            [
                self.cfg.hand_contact_height_threshold,
                self.cfg.hand_contact_height_threshold,
                self.cfg.knee_contact_height_threshold,
                self.cfg.knee_contact_height_threshold,
                self.cfg.foot_contact_height_threshold,
                self.cfg.foot_contact_height_threshold,
            ],
            dtype=torch.float32,
            device=self.device,
        )
        self._mass_view = getattr(self.robot, "root_physx_view", None)
        self._default_link_masses = None
        self._mass_randomization_enabled = False
        if self._mass_view is not None and hasattr(self._mass_view, "get_masses") and hasattr(self._mass_view, "set_masses"):
            try:
                self._default_link_masses = self._as_torch(self._mass_view.get_masses()).clone()
                self._mass_randomization_enabled = True
            except Exception:
                self._default_link_masses = None
                self._mass_randomization_enabled = False
        self._torque_curriculum_step = 0
        self._last_torque_scale: float | None = None
        self._nominal_joint_effort_limits_sim, self._runtime_actuator_effort_limits = self._capture_effort_limit_state()
        self._torque_curriculum_enabled = (
            self.cfg.physics_engine.lower() == "physx"
            and self._nominal_joint_effort_limits_sim is not None
            and hasattr(self.robot, "write_joint_effort_limit_to_sim")
        )

        self.actions = torch.zeros(self.num_envs, len(self._joint_ids), dtype=torch.float32, device=self.device)
        self.prev_actions = torch.zeros_like(self.actions)
        self.position_targets = self._default_joint_pos.clone()
        self.fixed_position_targets = self._default_fixed_joint_pos.clone()
        self._actor_frame_dim = 3 + 3 + 3 + len(self._joint_ids) + len(self._joint_ids) + len(self._joint_ids) + SPARSE_POSE_DIM + 2
        self._actor_history = torch.zeros(
            self.num_envs,
            self.cfg.actor_history_steps,
            self._actor_frame_dim,
            dtype=torch.float32,
            device=self.device,
        )

        self.command_generator = SparsePoseCommandGenerator(
            num_envs=self.num_envs,
            device=self.device,
            dt=self.cfg.sim.dt * self.cfg.decimation,
            mode=self.cfg.teleop_mode,
            dataset_path=self.cfg.teleop_dataset_path,
        )
        self.teleop_command = self.command_generator.step()
        if self._torque_curriculum_enabled:
            self._apply_torque_limit_curriculum(force=True)

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

    def _select_fixed_joint_columns(self, tensor: torch.Tensor) -> torch.Tensor:
        tensor = self._as_torch(tensor)
        if self._fixed_joint_ids_tensor.numel() == 0:
            shape = list(tensor.shape)
            shape[-1] = 0
            return tensor.new_zeros(shape)
        return torch.index_select(tensor, dim=-1, index=self._fixed_joint_ids_tensor)

    def _maybe_select_joint_columns(self, attr_name: str) -> torch.Tensor | None:
        tensor = getattr(self.robot.data, attr_name, None)
        if tensor is None:
            return None
        return self._select_joint_columns(tensor).clone()

    def _resolve_default_root_state(self) -> torch.Tensor:
        default_root_state = self._as_torch(self.robot.data.default_root_state)
        if default_root_state.ndim == 1:
            default_root_state = default_root_state.unsqueeze(0).repeat(self.num_envs, 1)
        return default_root_state.clone()

    def _expand_limit_value(self, value, joint_count: int) -> torch.Tensor | None:
        if value is None:
            return None
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device).flatten()
        if tensor.numel() == 0:
            return None
        if tensor.numel() == 1:
            return tensor.repeat(joint_count)
        if tensor.numel() % joint_count == 0:
            return tensor.view(-1, joint_count)[0].clone()
        if tensor.numel() != joint_count:
            raise ValueError(f"Expected {joint_count} effort-limit values, got {tensor.numel()}.")
        return tensor.clone()

    def _capture_effort_limit_state(self) -> tuple[torch.Tensor | None, dict[str, tuple[torch.Tensor | None, torch.Tensor | None]]]:
        actuators = getattr(self.robot, "actuators", None)
        if not isinstance(actuators, dict):
            return None, {}

        total_joint_count = self._as_torch(self.robot.data.default_joint_pos).shape[-1]
        solver_limits = torch.zeros(total_joint_count, dtype=torch.float32, device=self.device)
        solver_mask = torch.zeros(total_joint_count, dtype=torch.bool, device=self.device)
        runtime_defaults: dict[str, tuple[torch.Tensor | None, torch.Tensor | None]] = {}

        for name, actuator in actuators.items():
            joint_indices = getattr(actuator, "joint_indices", None)
            if joint_indices is None:
                continue
            joint_indices = torch.as_tensor(joint_indices, dtype=torch.long, device=self.device).flatten()
            if joint_indices.numel() == 0:
                continue

            base_effort = self._expand_limit_value(getattr(actuator, "effort_limit", None), joint_indices.numel())
            base_effort_sim = self._expand_limit_value(getattr(actuator, "effort_limit_sim", None), joint_indices.numel())
            if base_effort_sim is None and base_effort is not None:
                base_effort_sim = base_effort.clone()

            runtime_defaults[name] = (
                base_effort.clone() if base_effort is not None else None,
                base_effort_sim.clone() if base_effort_sim is not None else None,
            )

            if base_effort_sim is not None:
                solver_limits[joint_indices] = base_effort_sim
                solver_mask[joint_indices] = True

        if not torch.any(solver_mask[self._joint_ids_tensor]):
            return None, runtime_defaults
        return solver_limits[self._joint_ids_tensor].clone(), runtime_defaults

    def _current_torque_scale(self) -> float:
        curriculum_steps = max(int(self.cfg.torque_curriculum_steps), 1)
        progress = min(float(self._torque_curriculum_step) / float(curriculum_steps), 1.0)
        return float(
            self.cfg.torque_curriculum_initial_scale
            + (self.cfg.torque_curriculum_final_scale - self.cfg.torque_curriculum_initial_scale) * progress
        )

    def _scale_limit_value(self, value: torch.Tensor | None, scale: float):
        if value is None:
            return None
        scaled = value * scale
        if scaled.numel() == 1:
            return float(scaled.item())
        return scaled.clone()

    def _apply_torque_limit_curriculum(self, force: bool = False) -> None:
        if not self._torque_curriculum_enabled or self._nominal_joint_effort_limits_sim is None:
            return

        scale = self._current_torque_scale()
        if not force and self._last_torque_scale is not None and abs(scale - self._last_torque_scale) < 1.0e-6:
            return

        effort_limits_sim = (self._nominal_joint_effort_limits_sim * scale).unsqueeze(0).expand(self.num_envs, -1)
        self.robot.write_joint_effort_limit_to_sim(effort_limits_sim, joint_ids=self._joint_ids)

        actuators = getattr(self.robot, "actuators", None)
        if isinstance(actuators, dict):
            for name, actuator in actuators.items():
                defaults = self._runtime_actuator_effort_limits.get(name)
                if defaults is None:
                    continue
                base_effort, base_effort_sim = defaults
                if hasattr(actuator, "effort_limit") and base_effort is not None:
                    actuator.effort_limit = self._scale_limit_value(base_effort, scale)
                if hasattr(actuator, "effort_limit_sim") and base_effort_sim is not None:
                    actuator.effort_limit_sim = self._scale_limit_value(base_effort_sim, scale)

        self._last_torque_scale = scale

    def _get_root_planar_velocity(self) -> torch.Tensor:
        root_lin_vel = getattr(self.robot.data, "root_lin_vel_w", None)
        if root_lin_vel is None:
            root_lin_vel = self.robot.data.root_lin_vel_b
        return self._as_torch(root_lin_vel)[:, :2]

    def _get_root_linear_velocity_b(self) -> torch.Tensor:
        root_lin_vel_b = getattr(self.robot.data, "root_lin_vel_b", None)
        if root_lin_vel_b is not None:
            return self._as_torch(root_lin_vel_b)

        root_lin_vel_w = self._as_torch(self.robot.data.root_lin_vel_w)
        root_quat = quat_normalize(self._as_torch(self.robot.data.root_quat_w))
        return quat_apply(quat_conjugate(root_quat), root_lin_vel_w)

    def _get_root_linear_acceleration_b(self) -> torch.Tensor:
        body_acc_w = getattr(self.robot.data, "body_acc_w", None)
        if body_acc_w is None:
            body_acc_w = getattr(self.robot.data, "body_com_lin_acc_w", None)
        if body_acc_w is None:
            return torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        root_acc_w = self._as_torch(body_acc_w)[:, self._root_body_id, :3]
        root_quat = quat_normalize(self._as_torch(self.robot.data.root_quat_w))
        return quat_apply(quat_conjugate(root_quat), root_acc_w)

    def _get_body_linear_velocity_w(self) -> torch.Tensor:
        body_vel_w = getattr(self.robot.data, "body_lin_vel_w", None)
        if body_vel_w is not None:
            return self._as_torch(body_vel_w)

        body_state_w = getattr(self.robot.data, "body_state_w", None)
        if body_state_w is not None:
            return self._as_torch(body_state_w)[..., 7:10]

        return torch.zeros((self.num_envs, len(self.robot.body_names), 3), dtype=torch.float32, device=self.device)

    def _compute_contact_features(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        body_pos_w = self._as_torch(self.robot.data.body_pos_w)
        body_vel_w = self._get_body_linear_velocity_w()

        contact_heights = body_pos_w[:, self._contact_body_ids, 2]
        contact_speeds = torch.linalg.norm(body_vel_w[:, self._contact_body_ids], dim=-1)
        contact_flags = (
            (contact_heights <= self._contact_height_thresholds.unsqueeze(0))
            & (contact_speeds <= self.cfg.contact_speed_threshold)
        ).float()
        return contact_flags, contact_heights, contact_speeds

    def _update_actor_history(self, actor_frame: torch.Tensor) -> torch.Tensor:
        self._actor_history = torch.roll(self._actor_history, shifts=-1, dims=1)
        self._actor_history[:, -1] = actor_frame
        return self._actor_history.reshape(self.num_envs, -1)

    def _build_actor_frame(self) -> torch.Tensor:
        joint_pos = self._select_joint_columns(self.robot.data.joint_pos)
        joint_vel = self._select_joint_columns(self.robot.data.joint_vel)
        root_ang_vel = self._as_torch(self.robot.data.root_ang_vel_b)
        projected_gravity = self._as_torch(self.robot.data.projected_gravity_b)
        root_lin_acc_b = self._get_root_linear_acceleration_b()
        joint_pos_scaled = self._scale_joint_pos(joint_pos)
        sparse_pose = self.teleop_command.flatten()
        phase_obs = torch.stack((torch.sin(self.teleop_command.phase), torch.cos(self.teleop_command.phase)), dim=-1)
        return torch.cat(
            (
                root_ang_vel,
                projected_gravity,
                root_lin_acc_b * self.cfg.root_lin_acc_scale,
                joint_pos_scaled,
                joint_vel * self.cfg.joint_vel_scale,
                self.prev_actions,
                sparse_pose,
                phase_obs,
            ),
            dim=-1,
        )

    def _build_critic_obs(self, actor_obs: torch.Tensor) -> torch.Tensor:
        root_lin_vel_b = self._get_root_linear_velocity_b()
        root_height = self._as_torch(self.robot.data.root_pos_w)[:, 2:3]
        contact_flags, contact_heights, contact_speeds = self._compute_contact_features()
        privileged = torch.cat((root_lin_vel_b, root_height, contact_flags, contact_heights, contact_speeds), dim=-1)
        return torch.cat((actor_obs, privileged), dim=-1)

    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        if self._torque_curriculum_enabled:
            self._torque_curriculum_step += 1
            self._apply_torque_limit_curriculum()
        self.prev_actions.copy_(self.actions)
        self.actions = actions.clone()
        unclipped_targets = self._default_joint_pos + self.cfg.action_scale * self.actions
        self.position_targets = torch.clamp(unclipped_targets, self._joint_lower, self._joint_upper)

    def _apply_action(self) -> None:
        self.robot.set_joint_position_target(self.position_targets, joint_ids=self._joint_ids)
        if self._fixed_joint_ids:
            self.robot.set_joint_position_target(self.fixed_position_targets, joint_ids=self._fixed_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        self.teleop_command = self.command_generator.step()
        actor_frame = self._build_actor_frame()
        actor_obs = self._update_actor_history(actor_frame)
        critic_obs = self._build_critic_obs(actor_obs)
        task_reward = getattr(
            self,
            "reward_buf",
            torch.zeros(self.num_envs, dtype=torch.float32, device=self.device),
        )
        self.extras = {
            "add_diff": self._compute_add_differential(),
            "task_reward": task_reward.clone(),
        }
        return {"policy": actor_obs, "critic": critic_obs}

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

        root_lin_vel_xy = self._get_root_planar_velocity()
        cmd_vel_xy = self.teleop_command.target_lin_vel_xy
        locomotion_error = torch.linalg.norm(root_lin_vel_xy - cmd_vel_xy, dim=-1)
        locomotion_reward = torch.exp(-4.0 * locomotion_error.square())

        projected_gravity = self._as_torch(self.robot.data.projected_gravity_b)
        upright_reward = torch.clamp((-projected_gravity[:, 2]), min=0.0, max=1.0)
        root_height = root_pos[:, 2]
        root_height_reward = torch.exp(-30.0 * (root_height - self._default_root_height).square())

        contact_flags, _, contact_speeds = self._compute_contact_features()
        foot_contact = contact_flags[:, -2:]
        foot_speed = contact_speeds[:, -2:]
        foot_slip_penalty = torch.sum(foot_contact * foot_speed.square(), dim=-1)

        action_rate_penalty = torch.sum((self.actions - self.prev_actions).square(), dim=-1)
        joint_pos = self._select_joint_columns(self.robot.data.joint_pos)
        joint_vel = self._select_joint_columns(self.robot.data.joint_vel)
        energy_penalty = torch.sum((self.actions * joint_vel).square(), dim=-1)
        root_acc_penalty = torch.sum(self._get_root_linear_acceleration_b().square(), dim=-1)
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
            + self.cfg.root_height_weight * root_height_reward
            - self.cfg.action_rate_weight * action_rate_penalty
            - self.cfg.energy_weight * energy_penalty
            - self.cfg.foot_slip_weight * foot_slip_penalty
            - self.cfg.root_acc_weight * root_acc_penalty
            - self.cfg.joint_limit_weight * joint_limit_penalty
        )

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        projected_gravity = self._as_torch(self.robot.data.projected_gravity_b)
        tilt_angle = torch.acos(torch.clamp(-projected_gravity[:, 2], -1.0, 1.0)).abs()
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        fallen = tilt_angle > self.cfg.termination_tilt_angle
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
        fixed_joint_pos = self._default_fixed_joint_pos[env_ids].clone()
        fixed_joint_vel = self._default_fixed_joint_vel[env_ids].clone()
        joint_pos += self.cfg.reset_joint_pos_noise * torch.randn_like(joint_pos)
        joint_pos = torch.clamp(joint_pos, self._joint_lower, self._joint_upper)

        default_root_state = self._default_root_state[env_ids].clone()
        default_root_state[:, :3] += self.scene.env_origins[env_ids]
        default_root_state[:, 0] += self.cfg.reset_xy_pos_noise * (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0)
        default_root_state[:, 1] += self.cfg.reset_xy_pos_noise * (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0)
        default_root_state[:, 2] += self.cfg.reset_z_pos_noise * (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0)

        yaw_noise = self.cfg.reset_yaw_noise * (2.0 * torch.rand(len(env_ids), device=self.device) - 1.0)
        yaw_quat = quat_from_euler_xyz(
            torch.zeros_like(yaw_noise),
            torch.zeros_like(yaw_noise),
            yaw_noise,
        )
        default_root_state[:, 3:7] = quat_normalize(quat_mul(default_root_state[:, 3:7], yaw_quat))
        default_root_state[:, 7:10] += self.cfg.reset_lin_vel_noise * torch.randn_like(default_root_state[:, 7:10])
        default_root_state[:, 10:13] += self.cfg.reset_ang_vel_noise * torch.randn_like(default_root_state[:, 10:13])

        self.robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self.robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self.robot.write_joint_state_to_sim(joint_pos, joint_vel, joint_ids=self._joint_ids, env_ids=env_ids)
        if self._fixed_joint_ids:
            self.robot.write_joint_state_to_sim(
                fixed_joint_pos,
                fixed_joint_vel,
                joint_ids=self._fixed_joint_ids,
                env_ids=env_ids,
            )
        self.robot.set_joint_position_target(joint_pos, joint_ids=self._joint_ids, env_ids=env_ids)
        if self._fixed_joint_ids:
            self.robot.set_joint_position_target(fixed_joint_pos, joint_ids=self._fixed_joint_ids, env_ids=env_ids)
        self._randomize_joint_gains(env_ids)
        self._randomize_body_masses(env_ids)

        self.actions[env_ids] = 0.0
        self.prev_actions[env_ids] = 0.0
        self.position_targets[env_ids] = joint_pos
        self.fixed_position_targets[env_ids] = fixed_joint_pos
        self._actor_history[env_ids] = 0.0
        self.command_generator.reset(env_ids)

    def _randomize_joint_gains(self, env_ids: torch.Tensor) -> None:
        if self._default_joint_stiffness is None or self._default_joint_damping is None:
            return
        if not hasattr(self.robot, "write_joint_stiffness_to_sim") or not hasattr(self.robot, "write_joint_damping_to_sim"):
            return

        min_scale, max_scale = self.cfg.joint_gain_scale_range
        scale = min_scale + (max_scale - min_scale) * torch.rand(
            (len(env_ids), len(self._joint_ids)),
            dtype=torch.float32,
            device=self.device,
        )
        stiffness = self._default_joint_stiffness[env_ids] * scale
        damping = self._default_joint_damping[env_ids] * scale
        self.robot.write_joint_stiffness_to_sim(stiffness, joint_ids=self._joint_ids, env_ids=env_ids)
        self.robot.write_joint_damping_to_sim(damping, joint_ids=self._joint_ids, env_ids=env_ids)

    def _randomize_body_masses(self, env_ids: torch.Tensor) -> None:
        if not self._mass_randomization_enabled or self._default_link_masses is None:
            return

        mass_device = self._default_link_masses.device
        env_ids_for_masses = env_ids.to(device=mass_device, dtype=torch.long)
        min_scale, max_scale = self.cfg.mass_scale_range
        mass_scale = min_scale + (max_scale - min_scale) * torch.rand(
            (len(env_ids), self._default_link_masses.shape[-1]),
            dtype=torch.float32,
            device=mass_device,
        )
        masses = self._default_link_masses.clone()
        masses[env_ids_for_masses] = self._default_link_masses[env_ids_for_masses] * mass_scale
        try:
            self._mass_view.set_masses(masses, indices=env_ids_for_masses)
        except Exception:
            try:
                self._mass_view.set_masses(masses.cpu(), indices=env_ids.cpu())
            except Exception:
                self._mass_randomization_enabled = False

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

        vel_diff = self.teleop_command.target_lin_vel_xy - self._get_root_planar_velocity()
        return torch.cat(
            (
                torch.cat(pos_diffs, dim=-1),
                torch.cat(rot_diffs, dim=-1),
                vel_diff,
            ),
            dim=-1,
        )
