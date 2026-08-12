import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg
from isaaclab.assets import ArticulationCfg

MOMA_LAB_DATA_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../data")
)

UIKA_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        asset_path=f"{MOMA_LAB_DATA_DIR}/Robots/uika/uika_description/urdf/uika_description.urdf",
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
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
        pos=(0.0, 0.0, 0.3473),
        # joint_pos:默认站立姿态
        joint_pos={
            ".*L_hip_joint": -0.7,
            ".*R_hip_joint": 0.7,
            "F.*_thigh_joint": -0.15,
            "R.*_thigh_joint": -0.15,
            ".*_calf_joint": 0.75,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*_joint"],
            effort_limit=23.5,# 最大输出扭矩
            saturation_effort=23.5,# 最大饱和扭矩
            velocity_limit=30.0,
            stiffness=25.0,# 刚度系数
            damping=0.5,# 阻尼系数
            friction=0.0,# 摩擦系数
        ),
    },
)