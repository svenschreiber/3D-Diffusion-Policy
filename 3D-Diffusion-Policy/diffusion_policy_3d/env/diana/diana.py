# from .mjrl.mujoco_env import MujocoEnv
# from gym.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import os
import numpy as np
from typing import Union, Dict
import mujoco

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
            **kwargs,
        ):
        utils.EzPickle.__init__(
            self,
            xml_file, 
            frame_skip, 
            default_camera_config, 
            **kwargs
        )

        curr_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoEnv.__init__(
            self, 
            os.path.join(curr_dir, "assets", xml_file), 
            frame_skip, 
            observation_space=None,
            default_camera_config=default_camera_config,
            **kwargs,
        )

        self.active_joints = np.arange(8, 16)
        self.initial_pos = np.array([-0.785, 0, 0, 1.52, 0, -1.52, 0, 0])

        # Observation Space
        # - 8 qpos (right arm)
        # - 8 qvel (right arm)
        # - 3 ee pos
        # - 3 goal pos
        # - 1 ee to goal distance
        obs_dim = (len(self.active_joints) * 2 + 7,)
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=obs_dim, dtype=np.float32)

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
        action[:8] = 0
        self.do_simulation(action, self.frame_skip)

        obs = self.get_obs()
        reward, reward_info = self.get_reward(obs)
        info = reward_info

        if self.render_mode == "human":
            self.render()
        return obs, reward, obs[-1] < 0.02, False, info

    def get_reward(self, obs):
        dist = obs[-1]
        ee_vel = obs[14]
        n_contacts = len(self.data.contact)

        reward_info = {
            "reward_dist": -1.0 * dist,
            "reward_ee_vel": -0.01 * ee_vel,
            "reward_contacts": -5.0 * n_contacts,
        }

        return sum(reward_info.values()), reward_info
    
    def reset_model(self):
        goal_offset = np.concatenate(
            [
                self.np_random.uniform(low=-0.75, high=-0.45, size=1),
                self.np_random.uniform(low=-0.7, high=-0.5, size=1),
                self.np_random.uniform(low=0.9, high=1.2, size=1),
            ]
        )

        self.model.site("target").pos = goal_offset
        mujoco.mj_forward(self.model, self.data)
        qpos = self.init_qpos
        qpos[8:16] = self.initial_pos
        qvel = self.init_qvel
        self.set_state(qpos, qvel)
        return self.get_obs()

    def get_obs(self):
        ee_pos = self.data.body("arm_r_link_7").xpos
        goal_pos = self.data.site("target").xpos
        return np.concatenate(
            [
                self.data.qpos.flatten()[self.active_joints],
                self.data.qvel.flatten()[self.active_joints],
                ee_pos,
                goal_pos,
                (np.linalg.norm(goal_pos - ee_pos),),
            ]
        )