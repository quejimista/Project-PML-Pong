import gymnasium as gym

# from preprocessing import *


NAME_ENV = "PongNoFrameskip-v4"

if __name__ == "__main__":
    # env = make_env(NAME_ENV)
    env = gym.make(NAME_ENV)
    print(env.action_space)