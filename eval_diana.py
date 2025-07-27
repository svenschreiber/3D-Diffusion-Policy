from diffusion_policy_3d.env import DianaEnv
from gymnasium.wrappers import TimeLimit
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

env = TimeLimit(DianaEnv(render_mode="human", frame_skip=1), max_episode_steps=200)

ckpt_dir = Path("data/checkpoints")
ckpts = list(ckpt_dir.iterdir())
latest_ckpt = sorted(ckpts)[-1]
print("Loading", latest_ckpt)
model = sb3.PPO.load(latest_ckpt, device="cpu")

obs, info = env.reset()
resets = 0
while resets < 1000:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
        resets += 1
env.close()