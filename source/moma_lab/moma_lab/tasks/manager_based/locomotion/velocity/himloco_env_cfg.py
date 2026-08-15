"""HIMLoco observation layouts for Moma Lab quadrupeds."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import moma_lab.tasks.manager_based.locomotion.velocity.mdp as mdp


@configclass
class HIMPolicyCfg(ObsGroup):
    """Single-step proprioceptive observation expected by HIMEstimator."""

    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
    )
    base_ang_vel = ObsTerm(
        func=mdp.base_ang_vel,
        scale=0.25,
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-0.2, n_max=0.2),
    )
    projected_gravity = ObsTerm(
        func=mdp.projected_gravity,
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-0.05, n_max=0.05),
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-0.01, n_max=0.01),
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        scale=0.05,
        clip=(-100.0, 100.0),
        noise=Unoise(n_min=-1.5, n_max=1.5),
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0))

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class HIMCriticCfg(HIMPolicyCfg):
    """Policy observation followed immediately by estimator velocity target."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, scale=2.0, clip=(-100.0, 100.0))
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
        noise=Unoise(n_min=-0.1, n_max=0.1),
    )

    def __post_init__(self):
        self.enable_corruption = True
        self.concatenate_terms = True


@configclass
class HIMObservationsCfg:
    policy: HIMPolicyCfg = HIMPolicyCfg()
    critic: HIMCriticCfg = HIMCriticCfg()


def configure_himloco_observations(env_cfg, joint_names: list[str]) -> None:
    """Install the invariant HIMLoco layout after a robot's normal config is built."""
    env_cfg.observations = HIMObservationsCfg()
    for group in (env_cfg.observations.policy, env_cfg.observations.critic):
        group.joint_pos.params["asset_cfg"].joint_names = joint_names
        group.joint_vel.params["asset_cfg"].joint_names = joint_names
