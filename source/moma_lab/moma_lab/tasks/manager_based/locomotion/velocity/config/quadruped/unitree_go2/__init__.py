"""Register the Unitree Go2 velocity-tracking tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
    },
)

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2-HIMLoco-v0",
    entry_point="moma_lab.envs:HimlocoManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2HIMEnvCfg",
        "himloco_rsl_rl_cfg": "moma_lab.himloco_cfg:Go2HIMRunnerCfg",
    },
)
