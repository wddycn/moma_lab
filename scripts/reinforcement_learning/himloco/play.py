# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Play a checkpoint trained with the local HIMLoco runner."""

import argparse
import os
import sys
import time

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Play a checkpoint trained with HIMLoco.")
parser.add_argument("--video", action="store_true", default=False, help="Record a video during playback.")
parser.add_argument("--video_length", type=int, default=500, help="Recorded video length in environment steps.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable Fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, required=True, help="Name of the registered HIMLoco task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Agent configuration registry entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Environment seed.")
parser.add_argument("--real-time", action="store_true", default=False, help="Try to run at real-time speed.")
cli_args.add_himloco_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

if args_cli.video:
    args_cli.enable_cameras = True

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym
import torch

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

import moma_lab.tasks  # noqa: F401
from moma_lab.himloco import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from moma_lab.himloco.config import HIMOnPolicyRunnerCfg
from moma_lab.himloco.utils import export_himloco_policy_as_jit


torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _disable_if_present(group, name):
    """Disable an optional manager term without depending on a specific task config."""
    if group is not None and hasattr(group, name):
        setattr(group, name, None)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Load a HIMLoco checkpoint and run policy inference."""
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    agent_cfg.device = env_cfg.sim.device

    # Evaluation should be repeatable and should not continue training curricula.
    if hasattr(env_cfg.observations, "policy"):
        env_cfg.observations.policy.enable_corruption = False
    _disable_if_present(getattr(env_cfg, "events", None), "randomize_apply_external_force_torque")
    _disable_if_present(getattr(env_cfg, "events", None), "push_robot")
    _disable_if_present(getattr(env_cfg, "events", None), "randomize_push_robot")
    _disable_if_present(getattr(env_cfg, "events", None), "external_force")
    _disable_if_present(getattr(env_cfg, "curriculum", None), "command_levels_lin_vel")
    _disable_if_present(getattr(env_cfg, "curriculum", None), "command_levels_ang_vel")

    terrain = getattr(env_cfg.scene, "terrain", None)
    if terrain is not None:
        terrain.max_init_terrain_level = None
        generator = getattr(terrain, "terrain_generator", None)
        if generator is not None:
            generator.num_rows = min(generator.num_rows, 5)
            generator.num_cols = min(generator.num_cols, 5)
            generator.curriculum = False

    log_root_path = os.path.abspath(os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length,
    )
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, load_optimizer=False)
    policy = runner.get_inference_policy(device=env.device)

    export_himloco_policy_as_jit(
        runner.alg.actor_critic,
        path=os.path.join(log_dir, "exported"),
        filename="policy.pt",
    )

    obs = env.get_observations()
    dt = env.unwrapped.step_dt
    timestep = 0
    print("[INFO]: Starting HIMLoco playback.")
    while simulation_app.is_running():
        start_time = time.time()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, _, _, _, _, _ = env.step(actions)

        if args_cli.video:
            timestep += 1
            if timestep >= args_cli.video_length:
                break
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
