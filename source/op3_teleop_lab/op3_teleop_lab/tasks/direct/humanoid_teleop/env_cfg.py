from __future__ import annotations

import os

import isaaclab.sim as sim_utils
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

from op3_teleop_lab.utils.physics import build_sim_cfg

from .constants import POSITION_CENTRIC_COMMAND_DIM
from .robot_profile import SparseHumanoidRobotProfile, get_action_joint_names

PHYSICS_DT = 0.002
POLICY_CONTROL_HZ = 50.0
POLICY_DECIMATION = int(round(1.0 / (POLICY_CONTROL_HZ * PHYSICS_DT)))
ACTOR_HISTORY_STEPS = 25
CONTACT_GROUP_COUNT = 6


def compute_actor_frame_dim(action_dim: int) -> int:
    return 3 + 3 + 3 + action_dim + action_dim + action_dim + POSITION_CENTRIC_COMMAND_DIM


def compute_actor_obs_dim(action_dim: int, history_steps: int) -> int:
    return compute_actor_frame_dim(action_dim) * history_steps


def compute_critic_obs_dim(action_dim: int, history_steps: int, contact_group_count: int = CONTACT_GROUP_COUNT) -> int:
    privileged_dim = 3 + 1 + contact_group_count + contact_group_count + contact_group_count
    return compute_actor_obs_dim(action_dim, history_steps) + privileged_dim


def compute_action_dim(profile: SparseHumanoidRobotProfile) -> int:
    return len(get_action_joint_names(profile))


def resolve_teleop_mode(default_mode: str) -> str:
    return os.environ.get("HUMANOID_TELEOP_MODE", os.environ.get("OP3_TELEOP_MODE", default_mode))


def resolve_teleop_dataset_path(default_path: str | None) -> str | None:
    return os.environ.get("HUMANOID_TELEOP_DATASET_PATH", os.environ.get("OP3_TELEOP_DATASET_PATH", default_path))


def build_default_terrain_cfg() -> TerrainImporterCfg:
    return TerrainImporterCfg(
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


def build_contact_sensor_cfg(profile: SparseHumanoidRobotProfile) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"/World/envs/env_.*/Robot/({profile.contact_sensor_body_regex()})",
        history_length=3,
        track_air_time=True,
        force_threshold=1.0,
    )


def build_default_sim_cfg(physics_engine: str) -> object:
    return build_sim_cfg(physics_engine=physics_engine, dt=PHYSICS_DT, render_interval=POLICY_DECIMATION)
