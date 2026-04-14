import importlib
import os
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import ActuatorNetMLPCfg, DCMotorCfg, ImplicitActuatorCfg, IdealPDActuatorCfg, DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

#"C:/Users/VR2/Desktop/IsaacLab_Robots/op3_IsaacLab/op3urdf2usd/op3/op3.usd"
#"C:/Users/VR2/Desktop/mjcf_op3/new_op3/new_op3/scene/scene.usd"

_OP3_USD_PATH = str(Path(__file__).resolve().parent / "op3_asset" / "new_op3.usd")


def resolve_op3_cfg() -> ArticulationCfg:
    """Return the active OP3 articulation config.

    If ``OP3_CFG_IMPORT`` is set, it must point to a config object using either
    ``package.module:ATTR`` or ``package.module.ATTR`` syntax.
    """

    import_target = os.environ.get("OP3_CFG_IMPORT")
    if not import_target:
        return OP3_CFG

    module_name: str
    attr_name: str
    if ":" in import_target:
        module_name, attr_name = import_target.split(":", 1)
    else:
        module_name, _, attr_name = import_target.rpartition(".")

    if not module_name or not attr_name:
        raise ValueError(
            "OP3_CFG_IMPORT must use 'package.module:ATTR' or 'package.module.ATTR' syntax."
        )

    module = importlib.import_module(module_name)
    cfg = getattr(module, attr_name)
    if not isinstance(cfg, ArticulationCfg):
        raise TypeError(f"Imported OP3 config '{import_target}' is not an ArticulationCfg: {type(cfg)!r}")
    return cfg

OP3_ACTUATOR_CFG = ActuatorNetMLPCfg(
    joint_names_expr=[".*hip.*", ".*knee.*", ".*ank.*"],
    network_file=r"C:\Users\VR2\Documents\op3_effort_actuator_net_v3.pt",
    pos_scale=-1.0,
    vel_scale=1.0,
    torque_scale=1.0,
    input_order="pos_vel",
    input_idx=[0, 1, 2],
    effort_limit=2.5,  # taken from spec sheet
    velocity_limit=5,  # taken from spec sheet
    saturation_effort=2.5,  # same as effort limit
)
"""
"l_hip_yaw": 0.0,
            "l_hip_roll": -0.04,
            "l_hip_pitch": -0.65,
            "l_knee": 1.38,
            "l_ank_pitch": 0.83,
            "l_ank_roll": -0.01,
            "r_hip_yaw": 0.0,
            "r_hip_roll": 0.04,
            "r_hip_pitch": 0.65,
            "r_knee": -1.38,
            "r_ank_pitch": -0.83,
            "r_ank_roll": 0.01,
            "l_sho_pitch" : 0.0,
            "l_sho_roll" : 0.0,
            "l_el" : 0.0,
            "r_sho_pitch" : 0.0,
            "r_sho_roll" : 0.0,
            "r_el" : 0.0,
            "head_pan" : 0.0,
            "head_tilt" : 0.0,


            armature=0.00795,
            friction=0.015,
            effort_limit=3.6,
            effort_limit_sim=3.6,
            velocity_limit=5,
            velocity_limit_sim=5,
            stiffness={
                "head_pan": 27.4,
                "head_tilt": 27.4,
            },
            damping={
                "head_pan": 0.923,
                "head_tilt": 0.923,
            }
"""
OP3_MINIMAL_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/VazPC/Documents/op3_no_arms/op3_no_arms.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=800.0,
            max_angular_velocity=800.0,
            max_depenetration_velocity=0.0025,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            #sleep_threshold=0.00,
            #stabilization_threshold=0.000,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        
        pos=(0.0, 0.0, 0.26),
        #rot=(0.9990482, 0, 0.0436194, 0),
        joint_pos={
                "l_hip_yaw": 0.0,
                "l_hip_roll": -0.04,
                "l_hip_pitch": -0.40,    # reduced from -0.65
                "l_knee": 0.80,          # reduced from 1.38
                "l_ank_pitch": 0.50,     # reduced from 0.83
                "l_ank_roll": 0.00,

                "r_hip_yaw": 0.0,
                "r_hip_roll": 0.04,
                "r_hip_pitch": 0.40,     # reduced from 0.65
                "r_knee": -0.80,         # reduced from -1.38
                "r_ank_pitch": -0.50,    # reduced from -0.83
                "r_ank_roll": 0.00,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.85,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee.*"],
            armature=0.007950833,
            friction=0.0149524,
            effort_limit=3.6,
            effort_limit_sim=3.6,
            velocity_limit=4.71239,
            velocity_limit_sim=4.71239,
            stiffness={
                ".*_hip_yaw": 27.3873599,
                ".*_hip_roll": 27.3873599,
                ".*_hip_pitch": 27.3873599,
                ".*_knee": 27.3873599,
            },
            damping={
                ".*_hip_yaw": 0.92255651,
                ".*_hip_roll": 0.92255651,
                ".*_hip_pitch": 0.92255651,
                ".*_knee": 0.92255651,
            },
            min_delay=1,
            max_delay=4,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ank_pitch", ".*_ank_roll"],
            armature=0.007950833,
            friction=0.0149524,
            effort_limit=3.6,
            effort_limit_sim=3.6,
            velocity_limit=4.71239,
            velocity_limit_sim=4.71239,
            stiffness={
                ".*_ank_pitch": 27.3873599,
                ".*_ank_roll": 27.3873599,
            },
            damping={
                ".*_ank_pitch": 0.92255651,
                ".*_ank_roll": 0.92255651,
            },
            min_delay=1,
            max_delay=4,
        ),
    },
)

"""
                "l_hip_yaw": 0.0,
                "l_hip_roll": -0.04,
                "l_hip_pitch": -0.40,    # reduced from -0.65
                "l_knee": 0.80,          # reduced from 1.38
                "l_ank_pitch": 0.50,     # reduced from 0.83
                "l_ank_roll": 0.00,

                "r_hip_yaw": 0.0,
                "r_hip_roll": 0.04,
                "r_hip_pitch": 0.40,     # reduced from 0.65
                "r_knee": -0.80,         # reduced from -1.38
                "r_ank_pitch": -0.50,    # reduced from -0.83
                "r_ank_roll": 0.00,

                "l_sho_pitch": 0.0,
                "l_sho_roll": 0.785398,
                "l_el": -0.785398,
                "r_sho_pitch": 0.0,
                "r_sho_roll": -0.785398,
                "r_el": 0.785398,
                "head_pan": 0.0,
                "head_tilt": 0.0
"""

OP3_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot/",
    spawn=sim_utils.UsdFileCfg(
        usd_path=_OP3_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=800.0,
            max_angular_velocity=800.0,
            max_depenetration_velocity=0.5,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            #sleep_threshold=0.00,
            #stabilization_threshold=0.000,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(

        pos=(0.0, 0.0, 0.26),
        #rot=(0.9990482, 0, 0.0436194, 0),
        joint_pos={

            "l_hip_yaw": 0.0,
            "l_hip_roll": -0.04,
            "l_hip_pitch": -0.40,    # reduced from -0.65
            "l_knee": 0.80,          # reduced from 1.38
            "l_ank_pitch": 0.50,     # reduced from 0.83
            "l_ank_roll": 0.00,

            "r_hip_yaw": 0.0,
            "r_hip_roll": 0.04,
            "r_hip_pitch": 0.40,     # reduced from 0.65
            "r_knee": -0.80,         # reduced from -1.38
            "r_ank_pitch": -0.50,    # reduced from -0.83
            "r_ank_roll": 0.00,

            "l_sho_roll": 0.785398,
            "l_el": -0.785398,
            "r_sho_pitch": 0.0,
            "r_sho_roll": -0.785398,
            "r_el": 0.785398,
            "head_pan": 0.0,
            "head_tilt": 0.0,

        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.85,
    actuators={
        "legs": DelayedPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee.*"],
            armature=0.007950833,
            friction=0.0149524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                ".*_hip_yaw": 32.3873599,
                ".*_hip_roll": 32.3873599,
                ".*_hip_pitch": 32.3873599,
                ".*_knee": 32.3873599,
            },
            damping={
                ".*_hip_yaw": 0.5225561,
                ".*_hip_roll": 0.5225561,
                ".*_hip_pitch": 0.5225561,
                ".*_knee": 0.5225561,
            },
            min_delay=0,
            max_delay=4,
        ),
        "feet": DelayedPDActuatorCfg(
            joint_names_expr=[".*_ank_pitch", ".*_ank_roll"],
            armature=0.007950833,
            friction=0.0149524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                ".*_ank_pitch": 32.3873599,
                ".*_ank_roll": 32.3873599,
            },
            damping={
                ".*_ank_pitch": 0.5225561,
                ".*_ank_roll": 0.5225561,
            },
            min_delay=0,
            max_delay=4,
        ),
        "arms": DelayedPDActuatorCfg(
            joint_names_expr=[".*_sho_pitch", ".*_sho_roll", ".*_el"],
            armature=0.007950833,
            friction=0.0149524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            
            stiffness={
                ".*_sho_pitch": 32.3873599,  # 27.3873599
                ".*_sho_roll": 32.3873599,
                ".*_el": 32.3873599,
            },
            damping={
                ".*_sho_pitch": 0.5225561,  # 0.52255651
                ".*_sho_roll": 0.5225561,
                ".*_el": 0.5225561,
            },
            min_delay=0,
            max_delay=4,
        ),
    
        "head": DelayedPDActuatorCfg(
            joint_names_expr=["head_pan", "head_tilt"],
            armature=0.007950833,
            friction=0.0149524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                "head_pan": 32.3873599,
                "head_tilt": 32.3873599,
            },
            damping={
                "head_pan": 0.5225561,
                "head_tilt": 0.5225561,
            },
            min_delay=0,
            max_delay=4,
        ),
    },
)

OP3_OLD_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="c:\\Users\\VR2\\Desktop\\IsaacLab_Robots\\op3\\op3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=800.0,
            max_angular_velocity=800.0,
            max_depenetration_velocity=0.0025,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            #sleep_threshold=0.00,
            #stabilization_threshold=0.000,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        
        pos=(0.0, 0.0, 0.26),
        #rot=(0.9990482, 0, 0.0436194, 0),
        joint_pos={
                "l_hip_yaw": 0.0,
                "l_hip_roll": -0.04,
                "l_hip_pitch": -0.40,    # reduced from -0.65
                "l_knee": 0.80,          # reduced from 1.38
                "l_ank_pitch": 0.50,     # reduced from 0.83
                "l_ank_roll": 0.00,

                "r_hip_yaw": 0.0,
                "r_hip_roll": 0.04,
                "r_hip_pitch": 0.40,     # reduced from 0.65
                "r_knee": -0.80,         # reduced from -1.38
                "r_ank_pitch": -0.50,    # reduced from -0.83
                "r_ank_roll": 0.00,

                "l_sho_pitch": 0.0,
                "l_sho_roll": 0.785398,
                "l_el": -0.785398,
                "r_sho_pitch": 0.0,
                "r_sho_roll": -0.785398,
                "r_el": 0.785398,
                "head_pan": 0.0,
                "head_tilt": 0.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee.*"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                ".*_hip_yaw": 27.3873599,
                ".*_hip_roll": 27.3873599,
                ".*_hip_pitch": 27.3873599,
                ".*_knee": 27.3873599,
            },
            damping={
                ".*_hip_yaw": 0.92255651,
                ".*_hip_roll": 0.92255651,
                ".*_hip_pitch": 0.92255651,
                ".*_knee": 0.92255651,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ank_pitch", ".*_ank_roll"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                ".*_ank_pitch": 27.3873599,
                ".*_ank_roll": 27.3873599,
            },
            damping={
                ".*_ank_pitch": 0.92255651,
                ".*_ank_roll": 0.92255651,
            },
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_sho_pitch", ".*_sho_roll", ".*_el"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            
            stiffness={
                ".*_sho_pitch": 27.3873599,  # 27.3873599
                ".*_sho_roll": 27.3873599,
                ".*_el": 27.3873599,
            },
            damping={
                ".*_sho_pitch": 0.92255651,  # 0.52255651
                ".*_sho_roll": 0.92255651,
                ".*_el": 0.92255651,
            },
        ),
    
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan", "head_tilt"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                "head_pan": 27.3873599,
                "head_tilt": 27.3873599,
            },
            damping={
                "head_pan": 0.92255651,
                "head_tilt": 0.92255651,
            },
        ),
    },
)


OP3_ACTUATOR_TUNER_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="/home/vr3/IsaacRobots/op3/new_op3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=300.0,
            max_angular_velocity=300.0,
            max_depenetration_velocity=0.5,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.004, rest_offset=0.0),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            #sleep_threshold=0.00,
            #stabilization_threshold=0.000,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        
        pos=(0.0, 0.0, 0.26),
        #rot=(0.9990482, 0, 0.0436194, 0),
        joint_pos={
                "l_hip_yaw": 0.0,
                "l_hip_roll": -0.04,
                "l_hip_pitch": -0.40,    # reduced from -0.65
                "l_knee": 0.80,          # reduced from 1.38
                "l_ank_pitch": 0.50,     # reduced from 0.83
                "l_ank_roll": 0.00,

                "r_hip_yaw": 0.0,
                "r_hip_roll": 0.04,
                "r_hip_pitch": 0.40,     # reduced from 0.65
                "r_knee": -0.80,         # reduced from -1.38
                "r_ank_pitch": -0.50,    # reduced from -0.83
                "r_ank_roll": 0.00,

                "l_sho_pitch": 0.0,
                "l_sho_roll": 0.0,
                "l_el": 0.0,
                "r_sho_pitch": 0.0,
                "r_sho_roll": 0.0,
                "r_el": 0.0,
                "head_pan": 0.0,
                "head_tilt": 0.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.95,
    actuators={
        "legs": IdealPDActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee.*"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                ".*_hip_yaw": 32.3873599,
                ".*_hip_roll": 32.3873599,
                ".*_hip_pitch": 32.3873599,
                ".*_knee": 32.3873599,
            },
            damping={
                ".*_hip_yaw": 0.52255651,
                ".*_hip_roll": 0.52255651,
                ".*_hip_pitch": 0.52255651,
                ".*_knee": 0.52255651,
            },
        ),
        "feet": IdealPDActuatorCfg(
            joint_names_expr=[".*_ank_pitch", ".*_ank_roll"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                ".*_ank_pitch": 32.3873599,
                ".*_ank_roll": 32.3873599,
            },
            damping={
                ".*_ank_pitch": 0.52255651,
                ".*_ank_roll": 0.52255651,
            },
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_sho_pitch", ".*_sho_roll", ".*_el"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            
            stiffness={
                ".*_sho_pitch": 32.3873599,  # 27.3873599
                ".*_sho_roll": 32.3873599,
                ".*_el": 32.3873599,
            },
            damping={
                ".*_sho_pitch": 0.52255651,  # 0.52255651
                ".*_sho_roll": 0.52255651,
                ".*_el": 0.52255651,
            },
        ),
    
        "head": IdealPDActuatorCfg(
            joint_names_expr=["head_pan", "head_tilt"],
            armature=0.005950833,
            friction=0.0109524,
            effort_limit=3.75,
            effort_limit_sim=3.75,
            velocity_limit=4.8,
            velocity_limit_sim=4.8,
            stiffness={
                "head_pan": 32.3873599,
                "head_tilt": 32.3873599,
            },
            damping={
                "head_pan": 0.52255651,
                "head_tilt": 0.52255651,
            },

        ),
    },
)

OP3_BAM_V2_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/VR2/Desktop/IsaacLab_Robots/op3_IsaacLab/op3_description/urdf/op3/op3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            #sleep_threshold=0.00,
            #stabilization_threshold=0.000,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        
        pos=(0.0, 0.0, 0.26),
        #rot=(0.9990482, 0, 0.0436194, 0),
        joint_pos={
                "l_hip_yaw": 0.0,
                "l_hip_roll": -0.04,
                "l_hip_pitch": -0.40,    # reduced from -0.65
                "l_knee": 0.80,          # reduced from 1.38
                "l_ank_pitch": 0.50,     # reduced from 0.83
                "l_ank_roll": -0.01,

                "r_hip_yaw": 0.0,
                "r_hip_roll": 0.04,
                "r_hip_pitch": 0.40,     # reduced from 0.65
                "r_knee": -0.80,         # reduced from -1.38
                "r_ank_pitch": -0.50,    # reduced from -0.83
                "r_ank_roll": 0.01,

                "l_sho_pitch": 0.0,
                "l_sho_roll": 0.0,
                "l_el": 0.0,
                "r_sho_pitch": 0.0,
                "r_sho_roll": 0.0,
                "r_el": 0.0,
                "head_pan": 0.0,
                "head_tilt": 0.0
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.85,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee.*"],
            armature=0.009189155,
            friction=0.01629554,
            effort_limit=2.42105,
            effort_limit_sim=2.42105,
            velocity_limit=4.71239,
            velocity_limit_sim=4.71239,
            stiffness={
                ".*_hip_yaw": 18.418399,
                ".*_hip_roll": 18.418399,
                ".*_hip_pitch": 18.418399,
                ".*_knee": 18.418399,
            },
            damping={
                ".*_hip_yaw": 0.45399,
                ".*_hip_roll": 0.45399,
                ".*_hip_pitch": 0.45399,
                ".*_knee": 0.45399,
            }
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ank_pitch", ".*_ank_roll"],
            armature=0.009189155,
            friction=0.01629554,
            effort_limit=2.42105,
            effort_limit_sim=2.42105,
            velocity_limit=4.71239,
            velocity_limit_sim=4.71239,
            stiffness={
                ".*_ank_pitch": 18.418399,
                ".*_ank_roll": 18.418399,
            },
            damping={
                ".*_ank_pitch": 0.45399,
                ".*_ank_roll": 0.45399,
            }
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_sho_pitch", ".*_sho_roll", ".*_el"],
            armature=0.009189155,
            friction=0.01629554,
            effort_limit=2.42105,
            effort_limit_sim=2.42105,
            velocity_limit=4.71239,
            velocity_limit_sim=4.71239,
            stiffness={
                ".*_sho_pitch": 18.418399,
                ".*_sho_roll": 18.418399,
                ".*_el": 18.418399,
            },
            damping={
                ".*_sho_pitch": 0.45399,
                ".*_sho_roll": 0.45399,
                ".*_el": 0.45399,
            }
        ),
    
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan", "head_tilt"],
            armature=0.009189155,
            friction=0.01629554,
            effort_limit=2.42105,
            effort_limit_sim=2.42105,
            velocity_limit=4.71239,
            velocity_limit_sim=4.71239,
            stiffness={
                "head_pan": 18.418399,
                "head_tilt": 18.418399,
            },
            damping={
                "head_pan": 0.45399,
                "head_tilt": 0.45399,
            }
        ),
    },
)



OP3_MJCF_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Root/",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/VR2/Desktop/mjcf_op3/test_op3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3000.0,
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            #sleep_threshold=0.005,
            #stabilization_threshold=0.0005,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        
        pos=(0.0, 0.0, 0.245),
        #rot=(0.9990482, 0, 0.0436194, 0),
        joint_pos={
            "l_hip_yaw": 0.0,
            "l_hip_roll": -0.04,
            "l_hip_pitch": -0.65,
            "l_knee": 1.38,
            "l_ank_pitch": 0.83,
            "l_ank_roll": -0.01,
            "r_hip_yaw": 0.0,
            "r_hip_roll": 0.04,
            "r_hip_pitch": 0.65,
            "r_knee": -1.38,
            "r_ank_pitch": -0.83,
            "r_ank_roll": 0.01,
            "l_sho_pitch" : 0.0,
            "l_sho_roll" : 0.0,
            "l_el" : 0.0,
            "r_sho_pitch" : 0.0,
            "r_sho_roll" : 0.0,
            "r_el" : 0.0,
            "head_pan" : 0.0,
            "head_tilt" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.85,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee.*"],
            armature=0.045,
            friction=0.03,
            effort_limit_sim=5.0,
            stiffness={
                ".*_hip_yaw": 21.1,
                ".*_hip_roll": 21.1,
                ".*_hip_pitch": 21.1,
                ".*_knee": 21.1,
            },
            damping={
                ".*_hip_yaw": 1.084,
                ".*_hip_roll": 1.084,
                ".*_hip_pitch": 1.084,
                ".*_knee": 1.084,
            }
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ank_pitch", ".*_ank_roll"],
            armature=0.045,
            friction=0.03,
            effort_limit_sim=5.0,
            stiffness={
                ".*_ank_pitch": 21.1,
                ".*_ank_roll": 21.1,
            },
            damping={
                ".*_ank_pitch": 1.084,
                ".*_ank_roll": 1.084,
            }
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*_sho_pitch", ".*_sho_roll", ".*_el"],
            armature=0.045,
            friction=0.03,
            effort_limit_sim=5.0,
            stiffness={
                ".*_sho_pitch": 21.1,
                ".*_sho_roll": 21.1,
                ".*_el": 21.1,
            },
            damping={
                ".*_sho_pitch": 1.084,
                ".*_sho_roll": 1.084,
                ".*_el": 1.084,
            }
        ),
    
        "head": ImplicitActuatorCfg(
            joint_names_expr=["head_pan", "head_tilt"],
            armature=0.045,
            friction=0.03,
            effort_limit_sim=5.0,
            stiffness={
                "head_pan": 21.1,
                "head_tilt": 21.1,
            },
            damping={
                "head_pan": 1.084,
                "head_tilt": 1.084,
            }
        ),
    },
)

"""
     joint_pos={
            "head_pan": 0,
            "head_tilt": 0,
            "l_ank_pitch": 0.83,
            "l_ank_roll": -0.01368,
            "l_el": 0,
            "l_hip_pitch": -0.654,
            "l_hip_roll": -0.03985,
            "l_hip_yaw": 0.0,
            ".*l_knee.*": 1.37885,
            "l_sho_pitch": 0,
            "l_sho_roll": 0,
            "r_ank_pitch": -0.83,
            "r_ank_roll": 0.01368,
            "r_el": 0,
            "r_hip_pitch": 0.654,
            "r_hip_roll": 0.03985,
            "r_hip_yaw": 0.0,
            ".*r_knee.*": -1.37885,
            "r_sho_pitch": 0,
            "r_sho_roll": 0,
        },

"""
OP3_ALT_CFG = ArticulationCfg(
    prim_path="/World/envs/env_.*/Robot",
    spawn=sim_utils.UsdFileCfg(
        usd_path="C:/Users/VR2/Desktop/IsaacLab_Robots/op3_IsaacLab/op3urdf2usd/op3/op3.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            enable_gyroscopic_forces=True,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=3000.0,
            max_depenetration_velocity=5.0,
        ),

        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=4,
            #sleep_threshold=0.005,
            #stabilization_threshold=0.0005,
        ),
        copy_from_source=False,
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        
        pos=(0.0, 0.0, 0.245),
        #rot=(0.9990482, 0, 0.0436194, 0),
        joint_pos={
            "l_hip_yaw": 0.0,
            "l_hip_roll": -0.04,
            "l_hip_pitch": -0.65,
            "l_knee": 1.38,
            "l_ank_pitch": 0.83,
            "l_ank_roll": -0.01,
            "r_hip_yaw": 0.0,
            "r_hip_roll": 0.04,
            "r_hip_pitch": 0.65,
            "r_knee": -1.38,
            "r_ank_pitch": -0.83,
            "r_ank_roll": 0.01,
            "l_sho_pitch" : 0.0,
            "l_sho_roll" : 0.0,
            "l_el" : 0.0,
            "r_sho_pitch" : 0.0,
            "r_sho_roll" : 0.0,
            "r_el" : 0.0,
            "head_pan" : 0.0,
            "head_tilt" : 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.85,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*_hip_yaw", ".*_hip_roll", ".*_hip_pitch", ".*_knee.*"],
            armature=0.01,
            friction=0.03,
            effort_limit=300,
            effort_limit_sim=300,
            velocity_limit=100.0,
            velocity_limit_sim=100.0,
            stiffness={
                ".*_hip_yaw": 150.0,
                ".*_hip_roll": 150.0,
                ".*_hip_pitch": 200.0,
                ".*_knee": 200.0,
            },
            damping={
                ".*_hip_yaw": 5.0,
                ".*_hip_roll": 5.0,
                ".*_hip_pitch": 5.0,
                ".*_knee": 5.0,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ank_pitch", ".*_ank_roll"],
            armature=0.01,
            friction=0.05,
            effort_limit=20,
            effort_limit_sim=20,
            stiffness={
                ".*_ank_pitch": 20,
                ".*_ank_roll": 20,
            },
            damping={
                ".*_ank_pitch": 2,
                ".*_ank_roll": 2,
            },
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[".*_sho_pitch", ".*_sho_roll", ".*_el"],
            effort_limit=4.1,
            velocity_limit=4.8,
            armature=0.035,
            friction=0.05,
            stiffness={
                ".*_sho_pitch": 6.25,
                ".*_sho_roll": 6.25,
                ".*_el": 6.25,
            },
            damping={
                ".*_sho_pitch": 1.95,
                ".*_sho_roll": 1.95,
                ".*_el": 1.95,
            },
            effort_limit_sim=4.1,
            velocity_limit_sim=4.8,
        ),
    
        "head": IdealPDActuatorCfg(
            joint_names_expr=["head_pan", "head_tilt"],
            effort_limit=4.1,
            velocity_limit=5.0,
            armature=0.25,
            friction=0.03,
            stiffness={
                "head_pan": 22,
                "head_tilt": 22,
            },
            damping={
                "head_pan": 11,
                "head_tilt": 0.8,
            },
            effort_limit_sim=4.1,
            velocity_limit_sim=5.0,
        ),
    },
)
