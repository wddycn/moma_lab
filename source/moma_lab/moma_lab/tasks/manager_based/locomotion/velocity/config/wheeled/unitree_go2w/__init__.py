"""Register the Unitree Go2W rough-terrain velocity task."""

import gymnasium as gym

from . import agents

gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Unitree-Go2W-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2WRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2WRoughPPORunnerCfg",
    },
)
