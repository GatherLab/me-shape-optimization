import gymnasium as gym
from stable_baselines3 import PPO

from gym_env import MEResonanceEnv


# env = make_vec_env("CartPole-v1")
env = MEResonanceEnv()

models_dir = "./me-shape-optimization/reinforcement_learning/models/PPO-24seg"
model_path = f"{models_dir}/26000"
model = PPO.load(model_path, env=env)

episodes = 1
for ep in range(episodes):
    obs, info = env.reset()
    done = False
    while not done:
        env.render()
        action, _states = model.predict(obs)
        obs, reward, done, _, info = env.step(action)

        # Take a random action
        # random_action = env.action_space.sample()

        print("action", action)
        print("reward", reward)
