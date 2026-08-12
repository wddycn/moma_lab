"""Register the Unitree Go2 velocity-tracking tasks."""

import gymnasium as gym

from . import agents


gym.register(
    id="RobotLab-Isaac-Velocity-Rough-Uika-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UikaRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UikaRoughPPORunnerCfg",
    },
)
