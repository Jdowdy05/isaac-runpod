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

    action_scale = 0.45
    joint_vel_scale = 0.05
    action_rate_weight = 0.02
    energy_weight = 2.0e-4
    pose_pos_weight = 1.75
    pose_rot_weight = 0.4
    locomotion_weight = 1.1
    upright_weight = 0.8
    alive_reward = 0.2
    joint_limit_weight = 1.0e-2
    pose_tracking_sigma = 12.0
    body_orientation_sigma = 6.0

    teleop_mode = "synthetic"
    teleop_dataset_path: str | None = None

    scene = InteractiveSceneCfg(num_envs=2048, env_spacing=3.0, replicate_physics=True)
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
    observation_space = 3 + 3 + action_space + action_space + action_space + SPARSE_POSE_DIM + 2
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
        self.observation_space = 3 + 3 + self.action_space + self.action_space + self.action_space + SPARSE_POSE_DIM + 2
        self.state_space = 0


@configclass
class OP3TeleopNewtonEnvCfg(OP3TeleopEnvCfg):
    physics_engine = "newton"
