from __future__ import annotations


def build_sim_cfg(physics_engine: str, dt: float, render_interval: int):
    """Build a PhysX SimulationCfg for Isaac Lab 2.3.2 / Isaac Sim."""

    if physics_engine.lower() != "physx":
        raise ValueError(
            "This repository now targets Isaac Lab 2.3.2 with Isaac Sim/PhysX only. "
            f"Unsupported physics engine: {physics_engine}"
        )

    from isaaclab.sim import PhysxCfg, RigidBodyMaterialCfg, SimulationCfg

    return SimulationCfg(
        dt=dt,
        render_interval=render_interval,
        physics_material=RigidBodyMaterialCfg(
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
            gpu_max_rigid_contact_count=2**20,
            gpu_max_rigid_patch_count=2**19,
        ),
    )
