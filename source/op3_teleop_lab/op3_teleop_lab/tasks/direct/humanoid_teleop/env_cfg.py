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


def validate_reward_weight_signs(cfg) -> None:
    reward_weights = (
        "alive_reward",
        "pose_pos_weight",
        "pose_rot_weight",
        "add_diff_reward_weight",
        "body_velocity_weight",
        "root_velocity_weight",
        "foot_air_time_reward_weight",
        "foot_orientation_weight",
        "upright_weight",
        "root_height_weight",
    )
    penalty_weights = (
        "termination_penalty",
        "action_rate_weight",
        "raw_action_excess_weight",
        "energy_weight",
        "foot_slip_weight",
        "root_acc_weight",
        "joint_limit_weight",
        "torque_penalty_weight",
        "torque_limit_penalty_weight",
    )

    for name in reward_weights:
        if hasattr(cfg, name) and float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative because reward assembly adds this term.")
    for name in penalty_weights:
        if hasattr(cfg, name) and float(getattr(cfg, name)) < 0.0:
            raise ValueError(f"{name} must be non-negative because reward assembly subtracts this term.")


def _embodiment_env_name(embodiment: str | None, suffix: str) -> str | None:
    if embodiment is None:
        return None
    key = embodiment.strip().upper()
    if not key:
        return None
    return f"{key}_TELEOP_{suffix}"


def resolve_teleop_mode(default_mode: str, embodiment: str | None = None) -> str:
    specific_name = _embodiment_env_name(embodiment, "MODE")
    if "HUMANOID_TELEOP_MODE" in os.environ:
        return os.environ["HUMANOID_TELEOP_MODE"]
    if specific_name is not None and specific_name in os.environ:
        return os.environ[specific_name]
    return default_mode


def resolve_teleop_dataset_path(default_path: str | None, embodiment: str | None = None) -> str | None:
    specific_name = _embodiment_env_name(embodiment, "DATASET_PATH")
    if "HUMANOID_TELEOP_DATASET_PATH" in os.environ:
        return os.environ["HUMANOID_TELEOP_DATASET_PATH"]
    if specific_name is not None and specific_name in os.environ:
        return os.environ[specific_name]
    return default_path


def resolve_disable_env_add_diff_reward(default: bool = False, embodiment: str | None = None) -> bool:
    specific_name = _embodiment_env_name(embodiment, "DISABLE_ADD_DIFF_REWARD")
    value = os.environ.get("HUMANOID_DISABLE_ADD_DIFF_REWARD")
    if value is None and specific_name is not None:
        value = os.environ.get(specific_name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


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
