"""DC motor actuator with a per-environment command delay."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from isaaclab.actuators import DCMotor, DCMotorCfg
from isaaclab.utils import DelayBuffer, configclass
from isaaclab.utils.types import ArticulationActions


class DelayedDCMotor(DCMotor):
    """Delay position, velocity and effort commands by a randomized physics-step count."""

    cfg: "DelayedDCMotorCfg"

    def __init__(self, cfg: "DelayedDCMotorCfg", *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        if cfg.min_delay < 0 or cfg.max_delay < cfg.min_delay:
            raise ValueError("Expected 0 <= min_delay <= max_delay.")
        self.positions_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.velocities_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)
        self.efforts_delay_buffer = DelayBuffer(cfg.max_delay, self._num_envs, device=self._device)

    def reset(self, env_ids: Sequence[int] | None):
        super().reset(env_ids)
        num_envs = self._num_envs if env_ids is None or env_ids == slice(None) else len(env_ids)
        time_lags = torch.randint(
            self.cfg.min_delay,
            self.cfg.max_delay + 1,
            (num_envs,),
            dtype=torch.int,
            device=self._device,
        )
        for buffer in (self.positions_delay_buffer, self.velocities_delay_buffer, self.efforts_delay_buffer):
            buffer.set_time_lag(time_lags, env_ids)
            buffer.reset(env_ids)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        control_action.joint_positions = self.positions_delay_buffer.compute(control_action.joint_positions)
        control_action.joint_velocities = self.velocities_delay_buffer.compute(control_action.joint_velocities)
        control_action.joint_efforts = self.efforts_delay_buffer.compute(control_action.joint_efforts)
        return super().compute(control_action, joint_pos, joint_vel)


@configclass
class DelayedDCMotorCfg(DCMotorCfg):
    class_type: type = DelayedDCMotor
    min_delay: int = 0
    max_delay: int = 0
