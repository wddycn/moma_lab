# Copyright (c) 2024-2026 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def base_external_force(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return the composed persistent external force acting on the selected body."""
    asset: Articulation = env.scene[asset_cfg.name]
    return asset.permanent_wrench_composer.composed_force_as_torch[:, asset_cfg.body_ids, :].squeeze(1).clone()


def height_scan_clip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    clip: tuple[float, float] = (-1.0, 1.0),
    offset: float = 0.5,
) -> torch.Tensor:
    """Return terrain heights relative to the sensor, offset and clipped."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return torch.clip(height, clip[0], clip[1])


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor

def mid360_height_map(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    clip: tuple[float, float] = (-0.5, 0.5),
) -> torch.Tensor:
    """Return a fixed-size forward terrain elevation map.

    Positive values mean the terrain point is below the sensor reference;
    negative values mean it is closer/higher.
    """
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]

    # 使用机器人 base 高度作为参考，比使用 2 m 高的射线起点更合理。
    robot: Articulation = env.scene["robot"]
    base_z = robot.data.root_pos_w[:, 2].unsqueeze(1)

    hit_z = sensor.data.ray_hits_w[..., 2]
    relative_height = base_z - hit_z

    # 消除平地时的常量站立高度。
    # UIKA 默认 base 高度约为 0.33 m。
    terrain_feature = relative_height - 0.33

    # 没击中网格时 ray_hits 可能出现 inf。
    terrain_feature = torch.nan_to_num(
        terrain_feature,
        nan=clip[1],
        posinf=clip[1],
        neginf=clip[0],
    )

    return torch.clamp(terrain_feature, clip[0], clip[1])