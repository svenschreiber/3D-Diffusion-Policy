# from .mjrl.mujoco_env import MujocoEnv
# from gym.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import os
import numpy as np
from typing import Union, Dict

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class DianaEnv(MujocoEnv, utils.EzPickle):
    def __init__(
            self,
            xml_file: str = "setup_final_poncho.xml",
            frame_skip: int = 5,
            default_camera_config: Dict[str, Union[float, int]] = DEFAULT_CAMERA_CONFIG,
            reward_near_weight : float    = 1.0,
            reward_grip_weight : float    = 0.5,
            reward_dist_weight : float    = 2.0,
            reward_lift_weight : float    = 1.0,
            reward_control_weight : float = 0.001,
            **kwargs,
        ):
        utils.EzPickle.__init__(
            self,
            xml_file, 
            frame_skip, 
            default_camera_config, 
            reward_near_weight, 
            reward_grip_weight,
            reward_dist_weight, 
            reward_lift_weight,
            reward_control_weight, 
            **kwargs
        )

        self._reward_near_weight = reward_near_weight
        self._reward_grip_weight = reward_grip_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_lift_weight = reward_lift_weight
        self._reward_control_weight = reward_control_weight

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoEnv.__init__(
            self, 
            os.path.join(curr_dir, "assets", xml_file), 
            frame_skip, 
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        obs_dim = (self.model.nq + self.model.nv,)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float64)

        self.metadata = {
            "render_modes": [
                "human",
                "rgb_array",
                "depth_array",
                "rgbd_tuple",
            ],
            "render_fps": int(np.round(1.0 / self.dt)),
        }

    def step(self, action):
        self.do_simulation(action, self.frame_skip)

        observation = self._get_obs()
        reward, reward_info = self._get_rew(action)
        info = reward_info

        if self.render_mode == "human":
            self.render()
        return observation, reward, False, False, info

    def _get_rew(self, action):
        core_pos = self.get_body_com("cube")
        arm_r_pos = self.get_body_com("q_gripper_r_finger")
        goal_pos = self.get_body_com("goal")

        dist_core_arm = np.linalg.norm(core_pos - arm_r_pos)
        dist_core_goal = np.linalg.norm(core_pos - goal_pos)
        
        max_dist = 1.0

        reward_near = self._reward_near_weight * (max_dist - dist_core_arm)
        reward_dist = self._reward_dist_weight * (max_dist - dist_core_goal)
        reward_ctrl = -self._reward_control_weight * np.square(action).sum()

        gripper_pos = self.data.qpos[15]
        desired_closed_pos = 0.04
        reward_grip = -self._reward_grip_weight * abs(gripper_pos - desired_closed_pos)

        reward = reward_near + reward_dist + reward_ctrl + reward_grip

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_near": reward_near,
            "reward_grip": reward_grip,
        }

        return reward, reward_info
    
    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.array([0, 0])
        self.table_pos = self.get_body_com("table")
        while True:
            self.core_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.4, high=-0.1, size=1),
                    self.np_random.uniform(low=-0.1, high=0.1, size=1),
                ]
            )
            if np.linalg.norm(self.core_pos - self.goal_pos) > 0.05:
                break

        qpos[[18, 19]] = self.core_pos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten(),
                self.data.qvel.flatten(),
            ]
        )