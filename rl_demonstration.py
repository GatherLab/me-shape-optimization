import gym
from stable_baselines3 import PPO

import os

# from gymnasium import spaces
# import pygame
# import numpy as np

dir = "./img/RL/"
os.makedirs(dir, exist_ok=True)

env = gym.make("LunarLander-v2")  # , render_mode="human")
env.reset()

"""
model = PPO("MlpPolicy", env, verbose=1, tensorboard_log=dir)

timesteps = 10000

# This can also be just a while True loop
# total_timesteps = 50000
# for i in range(0, int(total_timesteps / timesteps)):
i = 0
while True:
    model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO")
    model.save(f"{dir}{i*timesteps}")
    i += 1

env.close()

"""
# Load back in the trained things
model_path = f"{dir}/5000000.zip"
model = PPO.load(model_path, env=env)

episodes = 10
for ep in range(episodes):
    obs = env.reset()
    done = False
    while not done:
        env.render()
        action, _ = model.predict(obs)
        obs, reward, done, info = env.step(action)
