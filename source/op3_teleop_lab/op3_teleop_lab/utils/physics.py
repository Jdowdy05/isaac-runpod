from __future__ import annotations

from typing import Any


def build_sim_cfg(physics_engine: str, dt: float, render_interval: int) -> Any:
    """Build an Isaac Lab SimulationCfg for PhysX or Newton.

    Newton imports are resolved lazily so the PhysX path still imports cleanly in
    environments without Newton installed.
    """

    from isaaclab.sim import SimulationCfg

    physics_engine = physics_engine.lower()
    if physics_engine == "physx":
        try:
            from isaaclab_physx.physics import PhysxCfg

            return SimulationCfg(
                dt=dt,
                render_interval=render_interval,
                physics=PhysxCfg(
                    bounce_threshold_velocity=0.2,
                ),
            )
        except ImportError:
            from isaaclab.sim import PhysxCfg

            return SimulationCfg(
                dt=dt,
                render_interval=render_interval,
                physics_material=None,
                physx=PhysxCfg(
                    bounce_threshold_velocity=0.2,
                    gpu_max_rigid_contact_count=2**20,
                    gpu_max_rigid_patch_count=2**19,
                ),
            )

    if physics_engine == "newton":
        try:
            from isaaclab_newton.physics import MJWarpSolverCfg, NewtonCfg

            return SimulationCfg(
                dt=dt,
                render_interval=render_interval,
                physics=NewtonCfg(
                    solver_cfg=MJWarpSolverCfg(
                        njmax=64,
                        nconmax=64,
                        ls_iterations=20,
                        cone="pyramidal",
                        ls_parallel=True,
                        integrator="implicitfast",
                        impratio=1.0,
                    ),
                    num_substeps=1,
                    debug_mode=False,
                ),
            )
        except ImportError as exc:
            try:
                from isaaclab.sim._impl.newton_manager_cfg import NewtonCfg
                from isaaclab.sim._impl.solvers_cfg import MJWarpSolverCfg

                return SimulationCfg(
                    dt=dt,
                    render_interval=render_interval,
                    newton_cfg=NewtonCfg(
                        solver_cfg=MJWarpSolverCfg(
                            nefc_per_env=64,
                            ls_iterations=20,
                            cone="pyramidal",
                            ls_parallel=True,
                            impratio=1,
                        ),
                        num_substeps=1,
                        debug_mode=False,
                    ),
                )
            except ImportError:
                raise ImportError(
                    "Newton support is not available in the current Isaac Lab install."
                ) from exc

    raise ValueError(f"Unsupported physics engine: {physics_engine}")
