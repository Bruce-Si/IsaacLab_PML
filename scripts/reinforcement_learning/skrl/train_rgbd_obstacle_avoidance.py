import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import datetime

# import the skrl components to build the RL system
from skrl.agents.torch.ppo import PPO, PPO_DEFAULT_CONFIG
from skrl.envs.loaders.torch import load_isaaclab_env
from skrl.envs.wrappers.torch import wrap_env
from skrl.memories.torch import RandomMemory
from skrl.models.torch import DeterministicMixin, GaussianMixin, Model
from skrl.resources.preprocessors.torch import RunningStandardScaler
from skrl.resources.schedulers.torch import KLAdaptiveRL
from skrl.trainers.torch import SequentialTrainer
from skrl.utils import set_seed


# seed for reproducibility
set_seed(41)  # e.g. `set_seed(42)` for fixed seed
now = datetime.datetime.now()
now_string = now.strftime('%Y_%m_%d_%H_%M')

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


# load and wrap the Isaac Lab environment
task_name = "Isaac-Quadcopter-RGBD-Obstacle-Avoidance-v1"
# env = load_isaaclab_env(task_name="Isaac-Quadcopter-RGBD-Obstacle-Avoidance-v0", num_envs=128)
env = load_isaaclab_env(task_name="Isaac-Quadcopter-RGBD-Obstacle-Avoidance-v1", num_envs=128)
env = wrap_env(env)

device = env.device


# instantiate a memory as rollout buffer (any memory can be used for this)
memory = RandomMemory(memory_size=24, num_envs=env.num_envs, device=device)


# instantiate the agent's models (function approximators).
# PPO requires 2 models, visit its documentation for more details
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#models
models = {}
# models["policy"] = Shared(env.observation_space, env.action_space, device)
models["policy"] = CNN_resnet10(env.observation_space, env.action_space, device)
models["value"] = models["policy"]  # same instance: shared model


# configure and instantiate the agent (visit its documentation to see all the options)
# https://skrl.readthedocs.io/en/latest/api/agents/ppo.html#configuration-and-hyperparameters
cfg = PPO_DEFAULT_CONFIG.copy()
cfg["rollouts"] = 24  # memory_size
cfg["learning_epochs"] = 5
cfg["mini_batches"] = 4  # 24 * 4096 / 24576
cfg["discount_factor"] = 0.99
cfg["lambda"] = 0.95
cfg["learning_rate"] = 1e-3
cfg["learning_rate_scheduler"] = KLAdaptiveRL
cfg["learning_rate_scheduler_kwargs"] = {"kl_threshold": 0.01}
cfg["random_timesteps"] = 0
cfg["learning_starts"] = 0
cfg["grad_norm_clip"] = 1.0
cfg["ratio_clip"] = 0.2
cfg["value_clip"] = 0.2
cfg["clip_predicted_values"] = True
cfg["entropy_loss_scale"] = 0.0
cfg["value_loss_scale"] = 1.0
cfg["kl_threshold"] = 0
cfg["rewards_shaper"] = None
cfg["time_limit_bootstrap"] = False
cfg["state_preprocessor"] = RunningStandardScaler
cfg["state_preprocessor_kwargs"] = {"size": env.observation_space, "device": device}
cfg["value_preprocessor"] = RunningStandardScaler
cfg["value_preprocessor_kwargs"] = {"size": 1, "device": device}
# logging to TensorBoard and write checkpoints (in timesteps)
cfg["experiment"]["write_interval"] = 600
cfg["experiment"]["checkpoint_interval"] = 6000
cfg["experiment"]["directory"] = "runs/torch/" + task_name
cfg["experiment"]["wandb"] = True
cfg["experiment"]["wandb_kwargs"] = {
    "entity": "siwufei",
    "project": "IsaacLab",
    "name": now_string + "_" + task_name,
}

agent = PPO(models=models,
            memory=memory,
            cfg=cfg,
            observation_space=env.observation_space,
            action_space=env.action_space,
            device=device)

# Load the checkpoint
# agent.load("./runs/torch/Isaac-Quadcopter-RGBD-Obstacle-Avoidance-v1/25-03-04_10-02-49-453529_PPO/checkpoints/agent_137400.pt")

# configure and instantiate the RL trainer
cfg_trainer = {"timesteps": 4800000, "headless": False}
trainer = SequentialTrainer(cfg=cfg_trainer, env=env, agents=agent)

# start training
trainer.train()
