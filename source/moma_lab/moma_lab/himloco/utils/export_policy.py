# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""TorchScript export for the combined HIMLoco estimator and actor."""

from __future__ import annotations

import copy
import os

import torch
import torch.nn.functional as F


class _HIMLocoPolicyExporter(torch.nn.Module):
    """Combine observation-history estimation and deterministic policy inference."""

    def __init__(self, actor_critic):
        super().__init__()
        self.encoder = copy.deepcopy(actor_critic.estimator.encoder)
        self.actor = copy.deepcopy(actor_critic.actor)
        self.num_one_step_obs = int(actor_critic.num_one_step_obs)

    def forward(self, obs_history: torch.Tensor) -> torch.Tensor:
        estimation = self.encoder(obs_history)
        velocity = estimation[..., :3]
        latent = F.normalize(estimation[..., 3:], dim=-1, p=2.0)
        current_obs = obs_history[..., : self.num_one_step_obs]
        return self.actor(torch.cat((current_obs, velocity, latent), dim=-1))


def export_himloco_policy_as_jit(actor_critic, path: str, filename: str = "policy.pt") -> str:
    """Export one deployable module mapping observation history directly to actions."""
    os.makedirs(path, exist_ok=True)
    exporter = _HIMLocoPolicyExporter(actor_critic).to("cpu").eval()
    scripted = torch.jit.script(exporter)
    output_path = os.path.join(path, filename)
    scripted.save(output_path)
    print(f"[INFO]: Exported HIMLoco TorchScript policy to: {output_path}")
    return output_path
