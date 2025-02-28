# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from omni.isaac.lab_tasks.utils.wrappers.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

# standup

@configclass
class AnymalCStandUpPoseRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 20
    max_iterations = 100000
    save_interval = 100
    experiment_name = "anymal_c_standup_pose_rough"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=0.5,
        actor_hidden_dims=[32, 32],
        critic_hidden_dims=[32, 32],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
    
@configclass
class AnymalCStandUpPoseFlatPPORunnerCfg(AnymalCStandUpPoseRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 100000
        self.experiment_name = "anymal_c_standup_pose_flat"
        self.policy.actor_hidden_dims = [32, 32]
        self.policy.critic_hidden_dims = [32, 32]