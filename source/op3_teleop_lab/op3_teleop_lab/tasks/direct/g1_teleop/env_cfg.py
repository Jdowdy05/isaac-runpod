from __future__ import annotations

import copy

from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_assets.robots.unitree import G1_29DOF_CFG

from op3_teleop_lab.tasks.direct.humanoid_teleop.env_cfg import (
    ACTOR_HISTORY_STEPS,
    PHYSICS_DT,
    POLICY_CONTROL_HZ,
    POLICY_DECIMATION,
    build_contact_sensor_cfg,
    build_default_sim_cfg,
    build_default_terrain_cfg,
    compute_action_dim,
    compute_actor_obs_dim,
    compute_critic_obs_dim,
    resolve_teleop_dataset_path,
    resolve_teleop_mode,
)

from .robot_profile import make_default_g1_profile

if POLICY_DECIMATION != 10:
    raise ValueError(
        "G1 teleop timing configuration is inconsistent: expected decimation 10 for 50 Hz policy control "
        f"with a {PHYSICS_DT:.3f} s physics timestep."
    )


def _make_g1_teleop_articulation_cfg():
    cfg = copy.deepcopy(G1_29DOF_CFG)
    cfg.prim_path = "/World/envs/env_.*/Robot"
    cfg.spawn.activate_contact_sensors = True
    cfg.init_state.rot = (0.0, 0.0, 0.0, 1.0)
    return cfg


@configclass
class G1TeleopEnvCfg(DirectRLEnvCfg):
    decimation = POLICY_DECIMATION
    episode_length_s = 20.0
    physics_engine = "physx"
    actor_history_steps = ACTOR_HISTORY_STEPS

    action_clip = 100.0
    joint_vel_scale = 0.05
    root_lin_acc_scale = 0.05
    action_rate_weight = 0.08
    raw_action_excess_weight = 1.0e-2
    energy_weight = 2.0e-4
    pose_pos_weight = 1.75
    pose_rot_weight = 0.4
    add_diff_reward_weight = 1.0
    add_diff_reward_sigma = 4.0
    upright_weight = 0.8
    root_height_weight = 0.4
    alive_reward = 0.2
    termination_penalty = 10.0
    joint_limit_weight = 1.0e-2
    foot_slip_weight = 5.0e-2
    root_acc_weight = 2.0e-3
    termination_tilt_angle = 1.2
    pose_tracking_sigma = 12.0
    body_orientation_sigma = 6.0

    mass_scale_range = (0.8, 1.2)
    joint_gain_scale_range = (0.5, 1.5)
    reset_xy_pos_noise = 0.05
    reset_z_pos_noise = 0.005
    reset_yaw_noise = 0.35
    reset_lin_vel_noise = 0.1
    reset_ang_vel_noise = 0.2
    reset_joint_pos_noise = 0.08
    torque_curriculum_initial_scale = 1.0
    torque_curriculum_final_scale = 1.0
    torque_curriculum_steps = 1

    teleop_mode = "synthetic"
    teleop_dataset_path: str | None = None
    truncate_on_command_end = True

    scene = InteractiveSceneCfg(num_envs=2048, env_spacing=3.0, replicate_physics=True)
    terrain = build_default_terrain_cfg()
    sim = build_default_sim_cfg(physics_engine="physx")

    profile = make_default_g1_profile()
    robot = _make_g1_teleop_articulation_cfg()
    contact_sensor = build_contact_sensor_cfg(profile)

    action_space = compute_action_dim(profile)
    observation_space = compute_actor_obs_dim(action_space, actor_history_steps)
    critic_observation_space = compute_critic_obs_dim(action_space, actor_history_steps, len(profile.contact_segment_names))
    state_space = 0

    def __post_init__(self) -> None:
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()
        self.teleop_mode = resolve_teleop_mode(self.teleop_mode)
        self.teleop_dataset_path = resolve_teleop_dataset_path(self.teleop_dataset_path)
        self.decimation = POLICY_DECIMATION
        self.sim = build_default_sim_cfg(self.physics_engine)
        self.contact_sensor = build_contact_sensor_cfg(self.profile)
        self.action_space = compute_action_dim(self.profile)
        self.observation_space = compute_actor_obs_dim(self.action_space, self.actor_history_steps)
        self.critic_observation_space = compute_critic_obs_dim(
            self.action_space, self.actor_history_steps, len(self.profile.contact_segment_names)
        )
        self.state_space = 0
