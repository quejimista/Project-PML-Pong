import numpy as np
import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import ale_py

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32) 
        #change the order of the matrix (frame). For using torch we need to swap the dimensions

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        obs, _, terminated, truncated, info = self.env.step(1)  # FIRE
        if terminated or truncated:
            obs, info = self.env.reset(**kwargs)
        return obs, info


class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    # change the range of values to [0,1]


def print_env_info(name, env):
    obs, _ = env.reset()
    print("\n\n*** {} Environment ***".format(name))
    print("Environment obs. : {}".format(env.observation_space.shape))
    print("Observation shape: {}, type: {} and range [{},{}]".format(obs.shape, obs.dtype, np.min(obs), np.max(obs)))
    # print("Observation sample:\n{}".format(obs))



def make_env(env_name):
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode=None) # get the environment
    print("Standard Env.        : {}".format(env.observation_space.shape)) 
    env = MaxAndSkipObservation(env, skip=4) # all frames too similar, then take one framework every 4
    print("MaxAndSkipObservation: {}".format(env.observation_space.shape))
    env = FireResetEnv(env) # starting some of the atari games
    env = ResizeObservation(env, (84, 84)) # define the 84x84 frames for the observations
    print("ResizeObservation    : {}".format(env.observation_space.shape))
    env = GrayscaleObservation(env, keep_dim=True) # convert observation to gray scale
    print("GrayscaleObservation : {}".format(env.observation_space.shape))
    env = ImageToPyTorch(env) # image to pytorch
    print("ImageToPyTorch       : {}".format(env.observation_space.shape))
    env = ReshapeObservation(env, (84, 84)) # remove the first dimensions
    print("ReshapeObservation   : {}".format(env.observation_space.shape))
    env = FrameStackObservation(env, stack_size=4) # stack the last four frames for keeping the dynamics
    print("FrameStackObservation: {}".format(env.observation_space.shape))
    env = ScaledFloatFrame(env) # scale the frames of the environment
    print("ScaledFloatFrame     : {}".format(env.observation_space.shape))
    
    return env




