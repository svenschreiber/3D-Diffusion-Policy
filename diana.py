from diffusion_policy_3d.env import DianaEnv
from gymnasium.wrappers import TimeLimit
import stable_baselines3 as sb3
from sb3_contrib import TQC

env = TimeLimit(DianaEnv(render_mode='human', frame_skip=5), max_episode_steps=2000)

# env = DianaEnv(render_mode='rgb_array')
# action = env.action_space.sample()
# env.step(action)
# print(env.data.site("target"))
# exit(-1)

model = sb3.PPO(
    policy="MlpPolicy",
    env=env,
    batch_size=512,
    policy_kwargs={'net_arch':[512, 512, 512]},
    learning_rate=0.0001,
    verbose=1,
)
# model = TQC(
#     policy='MlpPolicy',
#     env=env,
#     batch_size=512,
#     learning_rate=0.001,
#     gamma=0.98,
#     policy_kwargs={'net_arch':[512, 512, 512], 'n_critics':2},
#     tau=0.005,
#     verbose=1,
#     device='mps',
# )

model.learn(total_timesteps=10_000_000)

# model.save("dianaenv")

obs, info = env.reset()
for _ in range(1000):
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    if terminated or truncated:
        obs, info = env.reset()
env.close()