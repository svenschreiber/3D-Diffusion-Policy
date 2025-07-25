from diffusion_policy_3d.env import DianaEnv
from gymnasium.wrappers import TimeLimit
from stable_baselines3 import PPO

env = TimeLimit(DianaEnv(render_mode='human'), max_episode_steps=2000)

model = PPO(
    policy="MlpPolicy",
    env=env,
    verbose=1,
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=1024,
    gamma=0.99,
    device="cpu"
)

model.learn(total_timesteps=1_000_000)

# model.save("dianaenv")

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()