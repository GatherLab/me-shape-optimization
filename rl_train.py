# from stable_baselines3.common.env_checker import check_env
# from stable_baselines3.common.env_util import make_vec_env

from stable_baselines3 import PPO
from gym_env import MEResonanceEnv

import os


# env = make_vec_env("CartPole-v1")

models_dir = "./me-shape-optimization/reinforcement_learning/models/PPO-24seg"
logdir = "./me-shape-optimization/reinforcement_learning/logs"

if not os.path.exists(models_dir):
    os.makedirs(models_dir)

if not os.path.exists(logdir):
    os.makedirs(logdir)

env = MEResonanceEnv()
# models_dir = "./me-shape-optimization/models/"
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=logdir)

TIMESTEPS = 1000

iters = 0
while True:
    model.learn(
        total_timesteps=TIMESTEPS,
        reset_num_timesteps=False,
        tb_log_name="PPO-24seg-rew-freq",
    )
    model.save(f"{models_dir}/{TIMESTEPS*(i+1)}")

"""
model.learn(total_timesteps=10000)

episodes = 100

for episode in range(episodes):
    done = False
    obs = env.reset()
    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, info = env.step(action)

        # Take a random action
        # random_action = env.action_space.sample()

        print("action", action)
        print("reward", reward)
"""
