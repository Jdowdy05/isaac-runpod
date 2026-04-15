from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.envs import DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass

from op3_teleop_lab.assets.op3 import resolve_op3_cfg
from op3_teleop_lab.utils.physics import build_sim_cfg

from .constants import SPARSE_POSE_DIM
from .robot_profile import make_default_op3_profile

PHYSICS_DT = 0.002
POLICY_CONTROL_HZ = 50.0
POLICY_DECIMATION = int(round(1.0 / (POLICY_CONTROL_HZ * PHYSICS_DT)))
ACTOR_HISTORY_STEPS = 10
CONTACT_GROUP_COUNT = 6


def compute_actor_frame_dim(action_dim: int) -> int:
    return 3 + 3 + 3 + action_dim + action_dim + action_dim + SPARSE_POSE_DIM + 2


def compute_actor_obs_dim(action_dim: int, history_steps: int) -> int:
    return compute_actor_frame_dim(action_dim) * history_steps


def compute_critic_obs_dim(action_dim: int, history_steps: int) -> int:
    privileged_dim = 3 + 1 + CONTACT_GROUP_COUNT + CONTACT_GROUP_COUNT + CONTACT_GROUP_COUNT
    return compute_actor_obs_dim(action_dim, history_steps) + privileged_dim

if POLICY_DECIMATION != 10:
    raise ValueError(
        "OP3 teleop timing configuration is inconsistent: expected decimation 10 for 50 Hz policy control "
        "with a 0.002 s physics timestep."
    )


@configclass
class OP3TeleopEnvCfg(DirectRLEnvCfg):
    decimation = POLICY_DECIMATION
    episode_length_s = 20.0
    physics_engine = "physx"
    actor_history_steps = ACTOR_HISTORY_STEPS

    action_scale = 0.45
    joint_vel_scale = 0.05
    root_lin_acc_scale = 0.05
    action_rate_weight = 0.02
    energy_weight = 2.0e-4
    pose_pos_weight = 1.75
    pose_rot_weight = 0.4
    locomotion_weight = 1.1
    upright_weight = 0.8
    root_height_weight = 0.4
    alive_reward = 0.2
    joint_limit_weight = 1.0e-2
    foot_slip_weight = 5.0e-2
    root_acc_weight = 2.0e-3
    termination_tilt_angle = 1.2
    pose_tracking_sigma = 12.0
    body_orientation_sigma = 6.0
    foot_contact_height_threshold = 0.08
    knee_contact_height_threshold = 0.16
    hand_contact_height_threshold = 0.16
    contact_speed_threshold = 0.75

    mass_scale_range = (0.8, 1.2)
    joint_gain_scale_range = (0.5, 1.5)
    reset_xy_pos_noise = 0.05
    reset_z_pos_noise = 0.005
    reset_yaw_noise = 0.35
    reset_lin_vel_noise = 0.1
    reset_ang_vel_noise = 0.2
    reset_joint_pos_noise = 0.08

    teleop_mode = "synthetic"
    teleop_dataset_path: str | None = None

    scene = InteractiveSceneCfg(num_envs=4096, env_spacing=3.0, replicate_physics=True)
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=1.2,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    sim = build_sim_cfg(physics_engine="physx", dt=PHYSICS_DT, render_interval=decimation)

    profile = make_default_op3_profile()
    robot = resolve_op3_cfg().replace(prim_path="/World/envs/env_.*/Robot")

    action_space = len(profile.joint_names)
    observation_space = compute_actor_obs_dim(action_space, actor_history_steps)
    critic_observation_space = compute_critic_obs_dim(action_space, actor_history_steps)
    state_space = 0

    def __post_init__(self) -> None:
        super_post_init = getattr(super(), "__post_init__", None)
        if callable(super_post_init):
            super_post_init()
        self.teleop_mode = os.environ.get("OP3_TELEOP_MODE", self.teleop_mode)
        self.teleop_dataset_path = os.environ.get("OP3_TELEOP_DATASET_PATH", self.teleop_dataset_path)
        self.decimation = POLICY_DECIMATION
        self.sim = build_sim_cfg(self.physics_engine, dt=PHYSICS_DT, render_interval=self.decimation)
        self.action_space = len(self.profile.joint_names)
        self.observation_space = compute_actor_obs_dim(self.action_space, self.actor_history_steps)
        self.critic_observation_space = compute_critic_obs_dim(self.action_space, self.actor_history_steps)
        self.state_space = 0


@configclass
class OP3TeleopNewtonEnvCfg(OP3TeleopEnvCfg):
    physics_engine = "newton"
