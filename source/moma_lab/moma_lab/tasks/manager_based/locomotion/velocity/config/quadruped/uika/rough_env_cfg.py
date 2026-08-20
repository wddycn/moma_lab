# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import math
from dataclasses import MISSING

import numpy as np
import scipy.interpolate as interpolate

import isaaclab.terrains as terrain_gen
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.terrains.height_field.hf_terrains_cfg import HfTerrainBaseCfg
from isaaclab.terrains.height_field.utils import height_field_to_mesh

import moma_lab.tasks.manager_based.locomotion.velocity.mdp as mdp
from moma_lab.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from moma_lab.assets.uika import UIKA_CFG  # isort: skip


# ============================================================
# 电机参数随机化范围
# 用于 Domain Randomization，提高 sim2real 鲁棒性
# ============================================================
UIKA_PACE_ACTUATOR_RANGES = {
    "hip": {
        "armature": (0.012499551, 0.014251016),
        "viscous_friction": (0.000871807, 0.002837807),
        "friction": (0.005748361, 0.018038273),
    },
    "thigh": {
        "armature": (0.012442659, 0.014217399),
        "viscous_friction": (0.001674354, 0.003362268),
        "friction": (0.009873092, 0.022234440),
    },
    "calf": {
        "armature": (0.022163186, 0.024222745),
        "viscous_friction": (0.001198500, 0.002290815),
        "friction": (0.007793665, 0.015262470),
    },
}


@height_field_to_mesh
def _pyramid_slope_with_noise(difficulty: float, cfg) -> np.ndarray:
    """UIKA_lab pyramid slope with a low-resolution random-noise overlay."""
    from isaaclab.terrains.height_field import hf_terrains

    height_field = hf_terrains.pyramid_sloped_terrain.__wrapped__(difficulty, cfg)
    amplitude = cfg.noise_amplitude_range[0] + difficulty * (
        cfg.noise_amplitude_range[1] - cfg.noise_amplitude_range[0]
    )
    width, length = height_field.shape
    width_low = int(width * cfg.horizontal_scale / cfg.downsampled_scale)
    length_low = int(length * cfg.horizontal_scale / cfg.downsampled_scale)
    height_min = int(-amplitude / cfg.vertical_scale)
    height_max = int(amplitude / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)
    values = np.arange(height_min, height_max + height_step, height_step)
    low = np.random.choice(values, (width_low, length_low))
    x = np.linspace(0, width * cfg.horizontal_scale, width_low)
    y = np.linspace(0, length * cfg.horizontal_scale, length_low)
    spline = interpolate.RectBivariateSpline(x, y, low)
    x_full = np.linspace(0, width * cfg.horizontal_scale, width)
    y_full = np.linspace(0, length * cfg.horizontal_scale, length)
    return height_field + np.rint(spline(x_full, y_full)).astype(np.int16)


@configclass
class UikaPyramidSlopeWithNoiseCfg(HfTerrainBaseCfg):
    function = _pyramid_slope_with_noise
    slope_range: tuple[float, float] = MISSING
    platform_width: float = 1.0
    inverted: bool = False
    noise_amplitude_range: tuple[float, float] = MISSING
    noise_step: float = MISSING
    downsampled_scale: float = MISSING


@height_field_to_mesh
def _discrete_obstacles(difficulty: float, cfg) -> np.ndarray:
    """UIKA_lab discrete rectangular-obstacle height field."""
    max_height = cfg.max_height_range[0] + difficulty * (cfg.max_height_range[1] - cfg.max_height_range[0])
    max_height_units = int(max_height / cfg.vertical_scale)
    min_size = int(cfg.obstacle_size_range[0] / cfg.horizontal_scale)
    max_size = int(cfg.obstacle_size_range[1] / cfg.horizontal_scale)
    platform_size = int(cfg.platform_width / cfg.horizontal_scale)
    width = int(cfg.size[0] / cfg.horizontal_scale)
    length = int(cfg.size[1] / cfg.horizontal_scale)
    height_field = np.zeros((width, length), dtype=np.int16)
    heights = [-max_height_units, -max_height_units // 2, max_height_units // 2, max_height_units]
    widths = list(range(min_size, max_size, 4))
    lengths = list(range(min_size, max_size, 4))
    for _ in range(cfg.num_obstacles):
        if not widths or not lengths:
            break
        obstacle_width = int(np.random.choice(widths))
        obstacle_length = int(np.random.choice(lengths))
        if width - obstacle_width <= 0 or length - obstacle_length <= 0:
            continue
        start_i = int(np.random.choice(range(0, width - obstacle_width, 4)))
        start_j = int(np.random.choice(range(0, length - obstacle_length, 4)))
        height_field[start_i : start_i + obstacle_width, start_j : start_j + obstacle_length] = np.random.choice(
            heights
        )
    x1, x2 = (width - platform_size) // 2, (width + platform_size) // 2
    y1, y2 = (length - platform_size) // 2, (length + platform_size) // 2
    height_field[x1:x2, y1:y2] = 0
    return height_field


@configclass
class UikaDiscreteObstaclesCfg(HfTerrainBaseCfg):
    function = _discrete_obstacles
    max_height_range: tuple[float, float] = MISSING
    obstacle_size_range: tuple[float, float] = MISSING
    num_obstacles: int = MISSING
    platform_width: float = 1.0


UIKA_COBBLESTONE_ROAD_CFG = terrain_gen.TerrainGeneratorCfg(
    size=(8.0, 8.0),
    border_width=25.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    difficulty_range=(0.0, 1.0),
    use_cache=True,
    sub_terrains={
        "hf_pyramid_slope": terrain_gen.HfPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=3.0, border_width=0.0
        ),
        "hf_pyramid_slope_inv": terrain_gen.HfInvertedPyramidSlopedTerrainCfg(
            proportion=0.05, slope_range=(0.0, 0.4), platform_width=3.0, border_width=0.0
        ),
        "hf_slope_with_noise": UikaPyramidSlopeWithNoiseCfg(
            proportion=0.2,
            slope_range=(0.0, 0.4),
            platform_width=3.0,
            border_width=0.0,
            noise_amplitude_range=(0.01, 0.08),
            noise_step=0.005,
            downsampled_scale=0.2,
        ),
        "pyramid_stairs": terrain_gen.MeshPyramidStairsTerrainCfg(
            proportion=0.3, step_height_range=(0.05, 0.15), step_width=0.30, platform_width=3.0, border_width=0.0
        ),
        "pyramid_stairs_inv": terrain_gen.MeshInvertedPyramidStairsTerrainCfg(
            proportion=0.3, step_height_range=(0.05, 0.15), step_width=0.30, platform_width=3.0, border_width=0.0
        ),
        "discrete_obstacles": UikaDiscreteObstaclesCfg(
            proportion=0.1,
            max_height_range=(0.05, 0.15),
            obstacle_size_range=(1.0, 2.0),
            num_obstacles=20,
            platform_width=3.0,
        ),
    },
)


@configclass
class UikaRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UIKA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.soft_joint_pos_limit_factor = 1.0
        self.scene.terrain.terrain_generator = UIKA_COBBLESTONE_ROAD_CFG
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.pattern_cfg.size = (0.3, 0.4)
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.physics_material.restitution = 0.0
        # mid360 配置
        self.scene.mid360_height_scanner.prim_path = (
            "{ENV_REGEX_NS}/Robot/mid360_link")
        self.scene.mid360_height_scanner.update_period = (
            self.decimation * self.sim.dt)

        self.sim.physx.solver_type = 1
        self.sim.physx.max_position_iteration_count = 4
        self.sim.physx.max_velocity_iteration_count = 0
        self.sim.physx.bounce_threshold_velocity = 0.5
        self.sim.physx.gpu_max_rigid_patch_count = 2**23
        self.sim.physx.gpu_max_rigid_contact_count = 2**23

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-1.0, 1.0),
                "y": (-1.0, 1.0),
                "z": (0.0, 0.0),
                "roll": (-0.3, 0.3),
                "pitch": (-0.3, 0.3),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "z": (-0.2, 0.2),
                "roll": (-0.05, 0.05),
                "pitch": (-0.05, 0.05),
                "yaw": (0.0, 0.0),
            },
        }
        self.events.randomize_rigid_body_material.params.update(
            {
                "static_friction_range": (0.2, 1.25),
                "dynamic_friction_range": (0.2, 1.25),
                "restitution_range": (0.0, 0.0),
            }
        )
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 2.0)
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)
        self.events.randomize_reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_pace_hip = EventTerm(
            func=mdp.randomize_actuator_group_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_hip_joint"),
                **UIKA_PACE_ACTUATOR_RANGES["hip"],
            },
        )
        self.events.randomize_pace_thigh = EventTerm(
            func=mdp.randomize_actuator_group_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_thigh_joint"),
                **UIKA_PACE_ACTUATOR_RANGES["thigh"],
            },
        )
        self.events.randomize_pace_calf = EventTerm(
            func=mdp.randomize_actuator_group_parameters,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*_calf_joint"),
                **UIKA_PACE_ACTUATOR_RANGES["calf"],
            },
        )
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.external_force = EventTerm(
            func=mdp.apply_periodic_external_force_torque,
            mode="interval",
            interval_range_s=(0.02, 0.02),
            params={
                "period_step": 8,
                "force_range": (-30.0, 30.0),
                "torque_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
            },
        )
        self.events.randomize_push_robot.interval_range_s = (16.0, 16.0)
        self.events.randomize_push_robot.params["velocity_range"] = {"x": (-1.0, 1.0), "y": (-1.0, 1.0)}
        self.events.randomize_push_robot.params["asset_cfg"] = SceneEntityCfg(
            "robot", body_names=[self.base_link_name]
        )

        # ------------------------------Rewards------------------------------
        # episode 终止不额外扣分
        self.rewards.is_terminated.weight = 0

        # 惩罚机身上下运动
        self.rewards.lin_vel_z_l2.weight = -2.0
        # 惩罚 roll/pitch 方向角速度
        self.rewards.ang_vel_xy_l2.weight = -0.05
        # 惩罚机身倾斜
        self.rewards.flat_orientation_l2.weight = 0.0
        # 保持机身高度约 0.33 m
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        # 惩罚机身剧烈加速
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # ---------------- 关节惩罚 ----------------
        # 惩罚过大的关节力矩
        self.rewards.joint_torques_l2.weight = -2.5e-5
        # 不惩罚关节速度
        self.rewards.joint_vel_l2.weight = 0
        # 惩罚关节加速度
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # 惩罚关节碰到位置限制
        self.rewards.joint_pos_limits.weight = -5.0
        # 不惩罚速度限制
        self.rewards.joint_vel_limits.weight = 0
        # 惩罚机械功率消耗
        self.rewards.joint_power.weight = -2e-5
        # 有静止指令时，如果乱动则惩罚
        self.rewards.stand_still.weight = -2.0
        # 默认关节位置惩罚
        self.rewards.joint_pos_penalty.weight = -0.3#-0.3
        self.rewards.joint_pos_penalty.params["stand_still_scale"] = 16.0
        # 对角腿关节对称约束
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(thigh|calf).*", "RL_(thigh|calf).*"],
            ["FL_(thigh|calf).*", "RR_(thigh|calf).*"],
        ]
        self.rewards.joint_mirror.params["use_default_offset"] = True

        # ---------------- 动作平滑 ----------------
        # 惩罚前后两步 action 变化太大
        self.rewards.action_rate_l2.weight = -0.01

        # ---------------- 接触相关 ----------------
        # 除脚以外的部位接触地面会被惩罚
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        # 惩罚脚部过大的接触力
        self.rewards.contact_forces.weight = -1e-3
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # ---------------- 速度跟踪奖励 ----------------
        # 奖励机器人跟踪目标 xy 线速度
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        # 奖励机器人跟踪目标 yaw 角速度
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # ---------------- 步态相关 ----------------
        # 奖励合理的腾空时间
        self.rewards.feet_air_time.weight = 0.1#1.0
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 惩罚四条腿腾空/接触时间差异
        self.rewards.feet_air_time_variance.weight = -1.0#-4.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 关闭脚接触数量奖励
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.prolonged_swing = RewTerm(
            func=mdp.prolonged_swing,
            weight=-1.5,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
                "max_swing_time": 0.60,
                "command_name": "base_velocity",
            },
        )
        # 无运动指令时奖励脚接触
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 无速度指令时，如果脚还腾空，则扣分
        self.rewards.feet_air_without_cmd = RewTerm(
            func=mdp.feet_air_without_cmd,
            weight=-2.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=self.foot_link_name),
                "command_name": "base_velocity",
                "command_threshold": 0.1,   # 指令小于 0.1 认为是“静止”
                "contact_threshold": 1.0,   # 接触力判断阈值
            },
        )
        # 关闭 stumble 惩罚
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        # 惩罚脚在地面打滑
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        # 不在 rough terrain 上使用固定世界 Z 脚高：它会把地形高度计入误差，
        # 在本任务中的实际惩罚量级约为 UIKA 参考运行的 4--5 倍。
        self.rewards.feet_height = None
        # 摆动脚相对身体高度约束
        self.rewards.feet_height_body.weight = -6.0     # 原先是2.5
        self.rewards.feet_height_body.params["target_height"] = -0.16   # 原先是-0.33，贴近地面
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        # 对角小跑步态同步奖励
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["std"] = math.sqrt(0.5)
        # 对角腿为一组：
        # FL + RR
        # FR + RL
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        # 奖励机器人保持朝上
        self.rewards.upward.weight = 0.5

        # 该项在当前 moma 接触时序下会产生数十倍于正常值的尖峰；先由
        # feet_air_time_variance 和 feet_gait 约束摆动相，避免淹没速度奖励。
        self.rewards.prolonged_swing = None

        # weight = 0 的奖励项直接删除
        # 可以减少不必要的计算
        if self.__class__.__name__ == "UikaRoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ====================================================
        # Terminations：训练终止条件
        # ====================================================
        # 关闭非法接触导致 episode 提前结束
        self.terminations.illegal_contact = None

        # ====================================================
        # Curriculum：课程学习
        # ====================================================
        # 不逐渐提高目标线速度难度
        self.curriculum.command_levels_lin_vel = None
        # 不逐渐提高目标角速度难度
        self.curriculum.command_levels_ang_vel = None

        # ====================================================
        # Commands：目标速度指令
        # ====================================================
        # 不使用相对 heading 控制
        self.commands.base_velocity.rel_heading_envs = 0.0
        # 不给 heading 目标，只给速度目标
        self.commands.base_velocity.heading_command = False


@configclass
class UikaHIMEnvCfg(UikaRoughEnvCfg):
    """Uika environment with the observation contract required by HIMLoco."""

    def __post_init__(self):
        super().__post_init__()
        from moma_lab.tasks.manager_based.locomotion.velocity.himloco_env_cfg import (
            configure_himloco_observations,
        )

        configure_himloco_observations(self, self.joint_names)
        self.disable_zero_weight_rewards()
