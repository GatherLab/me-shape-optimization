from stable_baselines3.common.env_checker import check_env
from environment import MEResonanceEnv


# env = MEResonanceEnv()
# It will check your custom environment and output additional warnings if needed
# check_env(env)


env = MEResonanceEnv()
episodes = 50

for episode in range(episodes):
    done = False
    obs = env.reset()
    while True:  # not done:
        env.render()
        random_action = env.action_space.sample()
        print("action", random_action)
        obs, reward, done, info = env.step(random_action)
        print("reward", reward)
