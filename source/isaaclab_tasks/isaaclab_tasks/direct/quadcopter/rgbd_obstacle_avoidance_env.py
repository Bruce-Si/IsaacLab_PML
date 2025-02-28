# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import cv2
import numpy as np

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg, TerrainGeneratorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.sensors import Camera, CameraCfg, TiledCamera, TiledCameraCfg
import isaaclab.terrains.height_field as hf_gen

##
# Pre-defined configs
##
from isaaclab_assets import CRAZYFLIE_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip



class RgbdObstacleAvoidanceEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env: RgbdObstacleAvoidanceEnv, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)


@configclass
class RgbdObstacleAvoidanceEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 30.0
    decimation = 2
    action_space = 4
 
    state_space = 0
    debug_vis = True

    ui_window_class_type = RgbdObstacleAvoidanceEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 100,
        render_interval=decimation,
        disable_contact_processing=True,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )

    # obstacle
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        max_init_terrain_level=None,
        terrain_type="generator",
        terrain_generator=TerrainGeneratorCfg(
            seed=41,
            size=(100.0, 10.0),
            # border_width=20.0,
            border_width=10.0,
            num_rows=1,
            num_cols=1,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                "obstacles": hf_gen.HfDiscreteObstaclesTerrainCfg(
                    size=(100.0, 10.0),
                    horizontal_scale=0.1,
                    vertical_scale=0.1,
                    border_width=0.0,
                    num_obstacles=40,
                    obstacle_height_mode="choice",
                    obstacle_width_range=(0.2, 1.0),
                    obstacle_height_range=(0.5, 5.0),
                    platform_width=1.5,
                )
            },
        ),
        # physics_material=sim_utils.RigidBodyMaterialCfg(
        #     friction_combine_mode="multiply",
        #     restitution_combine_mode="multiply",
        #     static_friction=1.0,
        #     dynamic_friction=1.0,
        #     restitution=0.0,
        # ),
        debug_vis=False,
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=128, env_spacing=2.5, replicate_physics=True)

    # robot
    robot: ArticulationCfg = CRAZYFLIE_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # camera
    tiled_camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="/World/envs/env_.*/Robot/body/front_camera",
        # prim_path="/World/envs/env_.*/Camera",
        update_period=0.1,
        height=224,
        width=224,
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
        ),
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.2), rot=(0.7071, 0.0, 0.0, 0.7071), convention="world"
        ),
        debug_vis = True
    )
    # tiled_camera: CameraCfg = CameraCfg(
    #     prim_path="/World/envs/env_.*/Robot/Camera",
    #     # prim_path="/World/envs/env_.*/Camera",
    #     update_period=0.1,
    #     height=112,
    #     width=112,
    #     data_types=["rgb", "distance_to_camera"],
    #     spawn=sim_utils.PinholeCameraCfg(
    #         focal_length=24.0, focus_distance=400.0, horizontal_aperture=20.955, clipping_range=(0.1, 20.0)
    #     ),
    #     offset=CameraCfg.OffsetCfg(
    #         pos=(0.0, 0.0, 0.2), rot=(0.7071, 0.0, 0.0, 0.7071), convention="world"
    #     ),
    # )

    observation_space = [tiled_camera.height + 1, tiled_camera.width, 1]
    # observation_space = 12

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.01
    distance_to_goal_reward_scale = 15.0


class RgbdObstacleAvoidanceEnv(DirectRLEnv):
    cfg: RgbdObstacleAvoidanceEnvCfg

    def __init__(self, cfg: RgbdObstacleAvoidanceEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)

        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                # "distance_to_obstacle",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("body")[0]
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self._tiled_camera = TiledCamera(self.cfg.tiled_camera)
        # self._tiled_camera = Camera(self.cfg.tiled_camera)
        self.scene.sensors["tiled_camera"] = self._tiled_camera

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):
        self._actions = actions.clone().clamp(-1.0, 1.0)
        self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

    def _apply_action(self):
        self._robot.set_external_force_and_torque(self._thrust, self._moment, body_ids=self._body_id)

    def _get_observations(self) -> dict:
        # desired_pos_b, _ = subtract_frame_transforms(
        #     self._robot.data.root_link_state_w[:, :3], self._robot.data.root_link_state_w[:, 3:7], self._desired_pos_w
        # )
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        # swf
        depth_img = self._tiled_camera.data.output["distance_to_camera"].clip(0.1, 20)
        state_feature = torch.cat(
            [
                # self._robot.data.root_com_lin_vel_b,
                # self._robot.data.root_com_ang_vel_b,
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                self._robot.data.projected_gravity_b,
                desired_pos_b,
            ],
            dim=-1,
        )
        state_feature = F.pad(state_feature, (0, depth_img.shape[2] - state_feature.shape[-1])).unsqueeze(1).unsqueeze(-1)
        obs = torch.cat(
            [
                depth_img,
                state_feature,
            ],
            dim=1,
        )
        observations = {"policy": obs}
        self.camera_obs = depth_img

        # cv2.destroyAllWindows()
        img_name = "Depth Image"
        cv2.namedWindow(img_name, cv2.WINDOW_AUTOSIZE)
        img = depth_img[0, :, :, 0].cpu().numpy()
        img_normalized = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
        img_uint8 = np.uint8(img_normalized)
        img_colored = cv2.applyColorMap(img_uint8, cv2.COLORMAP_VIRIDIS)
        font = cv2.FONT_HERSHEY_SIMPLEX
        # text = "step_num: " + str(infos[0]["step_num"])
        org = (5, 15)
        fontScale = 0.3
        color = (0, 0, 255)
        thickness = 1
        # cv2.putText(img_colored, text, org, font, fontScale, color, thickness)
        cv2.resizeWindow(img_name, img_colored.shape[1] * 1, img_colored.shape[0] * 1)
        cv2.imshow(img_name, img_colored)
        key = cv2.waitKey(1)
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # lin_vel = torch.sum(torch.square(self._robot.data.root_com_lin_vel_b), dim=1)
        # ang_vel = torch.sum(torch.square(self._robot.data.root_com_ang_vel_b), dim=1)
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        # distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_link_pos_w, dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 0.8)
        obstacle_safety = self.camera_obs.max() - self.camera_obs.mean(dim=(1,2)).squeeze(1)

        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            # "distance_to_obstacle": obstacle_safety * -0.1 * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    # def compute_reward_v6(self, done, info, action):
    #     reward = 0
    #     reward_reach = 100
    #     reward_crash = -200
    #     reward_outside_z = -100
    #     reward_outside_xy = 0
    #
    #     if not done:
    #         # get away from start reward
    #         distance_now = self.get_distance_to_goal_2d()
    #         delta_dis = self.previous_distance_from_des_point - distance_now
    #         reward_distance = 1 * delta_dis / (self.v_xy_max * self.dt)
    #         self.previous_distance_from_des_point = distance_now
    #
    #         # potential energy
    #         distance_cost = 0.1 * self.dynamic_model.state_norm[0] / 255.0
    #
    #         # action cost
    #         action_cost = 0
    #
    #         # yaw_rate cost
    #         yaw_speed_cost = 0.1 * abs(action[-1]) / self.dynamic_model.yaw_rate_max_rad
    #
    #         if self.dynamic_model.navigation_3d:
    #             # add action and z error cost
    #             v_z_cost = 0.1 * ((abs(action[1]) / self.dynamic_model.v_z_max) ** 2)
    #
    #             z_err_thred = 5
    #             z_err = abs(self.dynamic_model.state_raw[1])
    #             if z_err > z_err_thred:
    #                 z_err_cost = 0.10 * (z_err / z_err_thred)
    #             else:
    #                 z_err_cost = 0.05 * (z_err / z_err_thred)
    #
    #             action_cost += (v_z_cost + z_err_cost)
    #
    #         action_cost += yaw_speed_cost
    #
    #         # yaw error cost
    #         yaw_error = self.dynamic_model.state_raw[2]
    #         yaw_error_cost = 0.1 * abs(yaw_error / 180)
    #
    #         # obstacle cost
    #         obs_punish_soft = 3
    #         obs_punish_hard = 1
    #         if self.min_distance_to_obstacles > obs_punish_soft:
    #             obs_cost = 0
    #         elif obs_punish_hard < self.min_distance_to_obstacles <= obs_punish_soft:
    #             obs_cost = 1 * (obs_punish_soft - self.min_distance_to_obstacles) / (obs_punish_soft - obs_punish_hard)
    #         else:
    #             obs_cost = obs_punish_hard / self.min_distance_to_obstacles ** 1.5
    #             if self.min_distance_to_obstacles <= 0.5:
    #                 print(
    #                     "min dis to obstacle: {:.4f}m, punish: {:.3f}".format(self.min_distance_to_obstacles, obs_cost))
    #                 action_cost = 0
    #                 yaw_error_cost = 0
    #                 if reward_distance < 0:
    #                     reward_distance = 0
    #
    #         reward = reward_distance - obs_cost - action_cost - yaw_error_cost - distance_cost
    #
    #     else:
    #         if info["is_success"]:
    #             reward = reward_reach
    #         if info["is_crash"]:
    #             print("crash min dis to obstacle: {:.4f}m".format(self.min_distance_to_obstacles))
    #             reward = reward_crash
    #         if info["is_not_in_workspace"]:
    #             current_pose = self.dynamic_model.get_position()
    #             z = current_pose[2]
    #             if z <= self.work_space_z[0] or z >= self.work_space_z[1]:
    #                 reward = reward_outside_z
    #             else:
    #                 reward = reward_outside_xy
    #
    #     return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # died = torch.logical_or(
        #     self._robot.data.root_link_pos_w[:, 2] < 0.1, self._robot.data.root_link_pos_w[:, 2] > 2.0
        # )
        died = torch.logical_or(
            self._robot.data.root_pos_w[:, 2] < 0.1, self._robot.data.root_pos_w[:, 2] > 5.0
        )
        self.is_success = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1) <= 1.0
        self.is_crashed = self.camera_obs.min() <= 0.2
        self.is_not_in_workspace = died
        return died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        # final_distance_to_goal = torch.linalg.norm(
        #     self._desired_pos_w[env_ids] - self._robot.data.root_link_pos_w[env_ids], dim=1
        # ).mean()
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Set target position
        # self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)

        self._desired_pos_w[env_ids, :3] = torch.tensor([0, 10, 1.5]).to(self.device)
        self._desired_pos_w[env_ids, 0] += (env_ids - self.num_envs//2).float()
        # self._desired_pos_w[env_ids, 1] = 10
        # self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        # self._desired_pos_w[env_ids, 2] = 1.5

        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        # Set start position
        default_root_state[:, :3] = torch.tensor([0, -10, 1.5])
        default_root_state[:, 0] += (env_ids - self.num_envs//2).float()
        # default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        # self._robot.write_root_link_pose_to_sim(default_root_state[:, :7], env_ids)
        # self._robot.write_root_com_velocity_to_sim(default_root_state[:, 7:], env_ids)
        # self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
