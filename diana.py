from diffusion_policy_3d.env import DianaEnv
from gymnasium.wrappers import TimeLimit
import stable_baselines3 as sb3

env = TimeLimit(DianaEnv(render_mode='human'), max_episode_steps=2000)

env = DianaEnv(render_mode='rgb_array')
action = env.action_space.sample()
env.step(action)
print(env.data.geom("table"))
exit(-1)

model = sb3.PPO(
    policy="MlpPolicy",
    env=env,
    batch_size=1024,
    verbose=1,
)

model.learn(total_timesteps=10_000_000)

# model.save("dianaenv")

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()