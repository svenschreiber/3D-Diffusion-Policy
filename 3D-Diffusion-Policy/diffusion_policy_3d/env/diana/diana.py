# from .mjrl.mujoco_env import MujocoEnv
# from gym.utils import EzPickle
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium import utils
from gymnasium.spaces import Box
import os
import numpy as np

DEFAULT_CAMERA_CONFIG = {
    "trackbodyid": -1,
    "distance": 4.0,
}

class DianaEnv(MujocoEnv, utils.EzPickle):
    def __init__(
            self,
            xml_file: str = "setup_final_poncho.xml",
            frame_skip: int = 5,
            default_camera_config: dict[str, float | int] = DEFAULT_CAMERA_CONFIG,
            reward_near_weight: float = 0.5,
            reward_dist_weight: float = 1,
            reward_control_weight: float = 0.1,
            **kwargs,
        ):
        utils.EzPickle.__init__(
            xml_file, 
            frame_skip, 
            default_camera_config, 
            reward_near_weight, 
            reward_dist_weight, 
            reward_control_weight, 
            **kwargs
        )

        self._reward_near_weight = reward_near_weight
        self._reward_dist_weight = reward_dist_weight
        self._reward_control_weight = reward_control_weight

        observation_space = Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float64)
        curr_dir = os.path.dirname(os.path.abspath(__file__))
        MujocoEnv.__init__(
            self, 
            curr_dir+xml_file, 
            frame_skip, 
            observation_space=observation_space,
            default_camera_config=default_camera_config,
            **kwargs,
        )

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
        # truncation=False as the time limit is handled by the `TimeLimit` wrapper added during `make`
        return observation, reward, False, False, info
    
    def _get_rew(self, action):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1) * self._reward_near_weight
        reward_dist = -np.linalg.norm(vec_2) * self._reward_dist_weight
        reward_ctrl = -np.square(action).sum() * self._reward_control_weight

        reward = reward_dist + reward_ctrl + reward_near

        reward_info = {
            "reward_dist": reward_dist,
            "reward_ctrl": reward_ctrl,
            "reward_near": reward_near,
        }

        return reward, reward_info

    def reset_model(self):
        qpos = self.init_qpos

        self.goal_pos = np.asarray([0, 0])
        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.3, high=0, size=1),
                    self.np_random.uniform(low=-0.2, high=0.2, size=1),
                ]
            )
            if np.linalg.norm(self.cylinder_pos - self.goal_pos) > 0.17:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2:] = self.goal_pos
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.data.qpos.flatten()[:7],
                self.data.qvel.flatten()[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )