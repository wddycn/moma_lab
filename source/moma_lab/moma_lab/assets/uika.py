import os

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg

from moma_lab.assets.delayed_motor import DelayedDCMotorCfg

MOMA_LAB_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data")
)

_UIKA_MOTOR_LIMITS = {
    "hip": (17.0, 28.80),
    "thigh": (17.0, 28.80),
    "calf": (31.7, 15.43),
}


def _make_uika_actuator_cfg(joint_name: str, joint_type: str) -> DelayedDCMotorCfg:
    effort_limit, velocity_limit = _UIKA_MOTOR_LIMITS[joint_type]
    return DelayedDCMotorCfg(
        joint_names_expr=[joint_name],
        effort_limit=effort_limit,
        saturation_effort=effort_limit,
        velocity_limit=velocity_limit,
        stiffness=30.0,
        damping=1.5,
        friction=0.0,
        dynamic_friction=0.0,
        viscous_friction=0.0,
        armature=0.0042,
        min_delay=2,
        max_delay=3,
    )


UIKA_ACTUATORS = {
    f"{leg}_{joint_type}": _make_uika_actuator_cfg(f"{leg}_{joint_type}_joint", joint_type)
    for joint_type in ("hip", "thigh", "calf")
    for leg in ("FL", "FR", "RL", "RR")
}

UIKA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{MOMA_LAB_DATA_DIR}/Robots/uika/uika_description/urdf/uika_simple_collision.urdf",
        fix_base=False,
        merge_fixed_joints=False,
        replace_cylinders_with_capsules=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                stiffness=0,
                damping=0,
            )
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # pos[2]:默认站立高度
        pos=(0.0, 0.0, 0.33),
        # joint_pos:默认站立姿态
        joint_pos={
            ".*L_hip_joint": -0.75,
            ".*R_hip_joint": 0.75,
            ".*_thigh_joint": 0.05,
            ".*_calf_joint": 0.70,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators=UIKA_ACTUATORS,
)
