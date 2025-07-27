from diffusion_policy_3d.env import DianaEnv
from gymnasium.wrappers import TimeLimit
import stable_baselines3 as sb3
from stable_baselines3.common.vec_env import DummyVecEnv
from pathlib import Path

def make_env(num_envs):
    def _make_env():
        return TimeLimit(DianaEnv(
            render_mode="none", 
            frame_skip=1
            ), max_episode_steps=200)
    return DummyVecEnv([_make_env for _ in range(num_envs)])

vec_env = make_env(num_envs=32)

model = sb3.PPO(
    policy="MlpPolicy",
    env=vec_env,
    verbose=1,
    device="cpu",
)

ckpt_dir = Path("data/checkpoints")
ckpt_dir.mkdir(parents=True, exist_ok=True)

ckpt_interval = 1_000_000
target_steps = 50_000_000

steps = 0
while steps < target_steps:
    next_chunk = min(ckpt_interval, target_steps - steps)
    model.learn(total_timesteps=next_chunk, reset_num_timesteps=False)
    steps += next_chunk
    ckpt_path = ckpt_dir / f"diana_reach_{steps//1000}k.zip"
    model.save(ckpt_path)
    print("Checkpoint saved to", ckpt_path)

model.save(ckpt_dir / f"diana_reach_final.zip")
print("Final checkpoint saved to", ckpt_path)
