# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Script to train RL agent with skrl.

Visit the skrl documentation (https://skrl.readthedocs.io) to see the examples structured in
a more user-friendly way.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime

from isaaclab.app import AppLauncher

from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed

from typing import Optional, Sequence
import sys
from skrl import logger

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with skrl.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--checkpoint", type=str, default=None, help="Path to model checkpoint to resume training.")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
parser.add_argument(
    "--ml_framework",
    type=str,
    default="torch",
    choices=["torch", "jax", "jax-numpy"],
    help="The ML framework used for training the skrl agent.",
)
parser.add_argument(
    "--algorithm",
    type=str,
    default="PPO",
    choices=["AMP", "PPO", "IPPO", "MAPPO"],
    help="The RL algorithm used for training the skrl agent.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import random
from datetime import datetime

import skrl
from packaging import version

# check for minimum supported skrl version
SKRL_VERSION = "1.4.1"
if version.parse(skrl.__version__) < version.parse(SKRL_VERSION):
    skrl.logger.error(
        f"Unsupported skrl version: {skrl.__version__}. "
        f"Install supported version using 'pip install skrl>={SKRL_VERSION}'"
    )
    exit()

if args_cli.ml_framework.startswith("torch"):
    from skrl.utils.runner.torch import Runner
elif args_cli.ml_framework.startswith("jax"):
    from skrl.utils.runner.jax import Runner

from isaaclab_rl.skrl import SkrlVecEnvWrapper

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_pickle, dump_yaml

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils.hydra import hydra_task_config

# config shortcuts
algorithm = args_cli.algorithm.lower()
agent_cfg_entry_point = "skrl_cfg_entry_point" if algorithm in ["ppo"] else f"skrl_{algorithm}_cfg_entry_point"


class resblock(nn.Module):
    def __init__(self, ch_in, ch_out, stride=1, size=[112, 112]):
        super(resblock, self).__init__()

        self.conv1 = nn.Conv2d(ch_in, ch_out, 3, stride=stride, padding=1)
        self.ln1 = nn.LayerNorm([math.ceil(size[0] / stride), math.ceil(size[1] / stride)])
        self.conv2 = nn.Conv2d(ch_out, ch_out, 3, stride=1, padding=1)
        self.ln2 = nn.LayerNorm([math.ceil(size[0] / stride), math.ceil(size[1] / stride)])

        self.extra = nn.Sequential()
        if ch_in != ch_out and stride == 1:
            self.extra = nn.Sequential(
                nn.Conv2d(ch_in, ch_out, 1, 1),
                nn.LayerNorm([size[0], size[1]])
            )
        elif ch_in != ch_out and stride == 2:
            self.extra = nn.Sequential(
                nn.MaxPool2d(2, stride=stride),
                nn.Conv2d(ch_in, ch_out, 1, 1),
                nn.LayerNorm([math.ceil(size[0] / stride), math.ceil(size[1] / stride)])
            )

    def forward(self, x):
        out = F.relu(self.ln1(self.conv1(x)))
        out = self.ln2(self.conv2(out))

        x = self.extra(x)

        out = out + x
        out = F.relu(out)

        return out

# define shared model (stochastic and deterministic models) using mixins
class CNN_resnet10(GaussianMixin, DeterministicMixin, Model):
    def __init__(self, observation_space, action_space, device, clip_actions=False,
                 clip_log_std=True, min_log_std=-20, max_log_std=2, reduction="sum"):
        Model.__init__(self, observation_space, action_space, device)
        GaussianMixin.__init__(self, clip_actions, clip_log_std, min_log_std, max_log_std, reduction)
        DeterministicMixin.__init__(self, clip_actions)

        # input 224*224
        # self.features_extractor = nn.Sequential(nn.Conv2d(1, 8, 5, 2, 2),
        #                          nn.LayerNorm([112, 112]),
        #                          nn.ReLU(),
        #                          resblock(8, 16, stride=2, size=[112, 112]),
        #                          resblock(16, 32, stride=2, size=[56, 56]),
        #                          resblock(32, 64, stride=1, size=[28, 28]),
        #                          resblock(64, 64, stride=1, size=[28, 28]),
        #                          nn.Conv2d(64, 1, 1, 1),
        #                          nn.Flatten(start_dim=1))

        # input 112*112
        self.features_extractor = nn.Sequential(nn.Conv2d(1, 8, 3, 1, 1),
                                                nn.LayerNorm([112, 112]),
                                                nn.ReLU(),
                                                resblock(8, 16, stride=2, size=[112, 112]),
                                                resblock(16, 32, stride=2, size=[56, 56]),
                                                resblock(32, 64, stride=1, size=[28, 28]),
                                                resblock(64, 64, stride=1, size=[28, 28]),
                                                nn.Conv2d(64, 1, 1, 1),
                                                nn.Flatten(start_dim=1))

        self.state_extractor = nn.Sequential(nn.Flatten(start_dim=1))

        self.net = nn.Sequential(nn.Linear(784+12, 128),
                                 # nn.BatchNorm1d(128),
                                 nn.Tanh(),
                                 nn.Linear(128, 64),
                                 # nn.BatchNorm1d(64),
                                 nn.Tanh(),
                                 nn.Linear(64, 32),
                                 # nn.BatchNorm1d(32),
                                 nn.Tanh())

        self.mean_layer = nn.Linear(32, self.num_actions)
        self.log_std_parameter = nn.Parameter(torch.zeros(self.num_actions))

        self.value_layer = nn.Linear(32, 1)

    def act(self, inputs, role):
        if role == "policy":
            return GaussianMixin.act(self, inputs, role)
        elif role == "value":
            return DeterministicMixin.act(self, inputs, role)

    def compute(self, inputs, role):
        input = inputs["states"].reshape([-1, *self.observation_space.shape])
        cnn_feature = self.features_extractor(input.permute(0, 3, 1, 2)[:,:,:-1,:])
        state_feature = self.state_extractor(input.permute(0, 3, 1, 2)[:,:,-1,:])
        mix_feature = torch.cat([cnn_feature, state_feature[:, :12]], dim=1)
        if role == "policy":
            self._shared_output = self.net(mix_feature)
            return self.mean_layer(self._shared_output), self.log_std_parameter, {}
        elif role == "value":
            shared_output = self.net(mix_feature) if self._shared_output is None else self._shared_output
            self._shared_output = None
            return self.value_layer(shared_output), {}



@hydra_task_config(args_cli.task, agent_cfg_entry_point)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: dict):
    """Train with skrl agent."""
    # override configurations with non-hydra CLI arguments
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # multi-gpu training config
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
    # max iterations for training
    if args_cli.max_iterations:
        agent_cfg["trainer"]["timesteps"] = args_cli.max_iterations * agent_cfg["agent"]["rollouts"]
    agent_cfg["trainer"]["close_environment_at_exit"] = False
    # configure the ML framework into the global skrl variable
    if args_cli.ml_framework.startswith("jax"):
        skrl.config.jax.backend = "jax" if args_cli.ml_framework == "jax" else "numpy"

    # randomly sample a seed if seed = -1
    if args_cli.seed == -1:
        args_cli.seed = random.randint(0, 10000)

    # set the agent and environment seed from command line
    # note: certain randomization occur in the environment initialization so we set the seed here
    agent_cfg["seed"] = args_cli.seed if args_cli.seed is not None else agent_cfg["seed"]
    env_cfg.seed = agent_cfg["seed"]

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "skrl", agent_cfg["agent"]["experiment"]["directory"])
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Logging experiment in directory: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{algorithm}_{args_cli.ml_framework}"
    print(f"Exact experiment name requested from command line {log_dir}")
    if agent_cfg["agent"]["experiment"]["experiment_name"]:
        log_dir += f'_{agent_cfg["agent"]["experiment"]["experiment_name"]}'
    # set directory into agent config
    agent_cfg["agent"]["experiment"]["directory"] = log_root_path
    agent_cfg["agent"]["experiment"]["experiment_name"] = log_dir
    # update log_dir
    log_dir = os.path.join(log_root_path, log_dir)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)
    dump_pickle(os.path.join(log_dir, "params", "env.pkl"), env_cfg)
    dump_pickle(os.path.join(log_dir, "params", "agent.pkl"), agent_cfg)

    # get checkpoint path (to resume training)
    resume_path = retrieve_file_path(args_cli.checkpoint) if args_cli.checkpoint else None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv) and algorithm in ["ppo"]:
        env = multi_agent_to_single_agent(env)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wandb
    now = datetime.datetime.now()
    now_string = now.strftime('%Y_%m_%d_%H_%M')
    agent_cfg["agent"]["experiment"]["wandb_kwargs"] = {
    "entity": "siwufei",
    "project": "IsaacLab",
    "name": now_string + "_" + args_cli.task,
}

    # wrap around environment for skrl
    env = SkrlVecEnvWrapper(env, ml_framework=args_cli.ml_framework)  # same as: `wrap_env(env, wrapper="auto")`

    # configure and instantiate the skrl runner
    # https://skrl.readthedocs.io/en/latest/api/utils/runner.html
    runner = Runner(env, agent_cfg)

    models = {}
    models["policy"] = CNN_resnet10(env.observation_space, env.action_space, env.device)
    models["value"] = models["policy"]  # same instance: shared model

    memory = RandomMemory(memory_size= agent_cfg["agent"]["rollouts"], num_envs=env.num_envs, device=env.device)

    runner._agent = PPO(models=models,
            memory=memory,
            cfg=agent_cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=env.device)

    # load checkpoint (if specified)
    if resume_path:
        print(f"[INFO] Loading model checkpoint from: {resume_path}")
        runner.agent.load(resume_path)

    # run training
    runner.run()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
