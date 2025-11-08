import gymnasium as gym
from preprocessing import *


NAME_ENV = "PongNoFrameskip-v4"

if __name__ == "__main__":
    # env = make_env(NAME_ENV)
    # env = gym.make(NAME_ENV)
    # print(env.action_space)

    env = make_env(NAME_ENV)
    print_env_info("Wrapped", env)
    obs, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  #random sample for now
        obs, reward, done, truncated, info = env.step(action)
