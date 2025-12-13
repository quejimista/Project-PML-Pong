import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.wrappers import (
    FrameStackObservation, 
    ResizeObservation, 
    ReshapeObservation
)
import ale_py

class ColorReduction(ObservationWrapper):
    """
    Color reduction wrapper - converts RGB to 2D grayscale using mode 'B' (Blue channel).
    We output (H, W) instead of (H, W, 1) to ensure FrameStack creates (4, H, W).
    """
    def __init__(self, env):
        super().__init__(env)
        old_shape = env.observation_space.shape
        # Change from (H, W, 3) to (H, W) - Drop the channel dim entirely
        new_shape = (old_shape[0], old_shape[1])
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=new_shape,
            dtype=env.observation_space.dtype
        )
    
    def observation(self, obs):
        # Extract blue channel (mode='B') and drop the dimension
        return obs[:, :, 2]

class NormalizeObservation(ObservationWrapper):
    """
    Normalize observations to [0, 1] range.
    """
    def __init__(self, env, env_min=0, env_max=1):
        super().__init__(env)
        self.env_min = env_min
        self.env_max = env_max
        
        self.observation_space = gym.spaces.Box(
            low=env_min,
            high=env_max,
            shape=env.observation_space.shape,
            dtype=np.float32
        )
    
    def observation(self, obs):
        # Normalize from [0, 255] to [env_min, env_max]
        return obs.astype(np.float32) / 255.0 * (self.env_max - self.env_min) + self.env_min

class DtypeWrapper(ObservationWrapper):
    """
    Convert observation dtype.
    """
    def __init__(self, env, dtype):
        super().__init__(env)
        self.target_dtype = dtype
        self.observation_space = gym.spaces.Box(
            low=env.observation_space.low.min(),
            high=env.observation_space.high.max(),
            shape=env.observation_space.shape,
            dtype=dtype
        )
    
    def observation(self, obs):
        return obs.astype(self.target_dtype)

class EpisodeInfoWrapper(gym.Wrapper):
    """
    Track episode statistics and ensure they're properly logged.
    """
    def __init__(self, env):
        super().__init__(env)
        self.episode_reward = 0
        self.episode_length = 0

    def reset(self, **kwargs):
        self.episode_reward = 0
        self.episode_length = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.episode_reward += reward
        self.episode_length += 1
        
        # Add episode info when episode ends
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
        
        return obs, reward, terminated, truncated, info

def make_env(env_name="ALE/Pong-v5", render=None, verbose=False):
    """
    Create and wrap Pong environment.
    Final observation space: Box(0.0, 1.0, (4, 84, 84), float32)
    """
    gym.register_envs(ale_py)
    
    # Base environment
    env = gym.make(env_name, render_mode=render)
    
    if verbose:
        print(f"Original: {env.observation_space.shape}")
    
    # 1. Color reduction (to 2D)
    env = ColorReduction(env)
    
    # 2. Resize to 84x84
    env = ResizeObservation(env, (84, 84))
    
    # 3. Frame stack (4) -> Result will be (4, 84, 84) because input is (84, 84)
    # Gymnasium FrameStack stacks on the first dimension.
    env = FrameStackObservation(env, stack_size=4)
    
    # 4. Convert dtype to float32
    env = DtypeWrapper(env, np.float32)
    
    # 5. Normalize observations
    env = NormalizeObservation(env, env_min=0, env_max=1)
    
    # Note: We removed ReshapeObservation because FrameStack on 2D inputs 
    # naturally produces (4, 84, 84), which matches the project requirement.
    
    # Add episode info tracking
    env = EpisodeInfoWrapper(env)
    
    if verbose:
        print(f"Final Space: {env.observation_space}")
    
    return env

if __name__ == "__main__":
    # Simple test
    env = make_env(verbose=True)
    print(f"Observation Space: {env.observation_space}")
    obs, _ = env.reset()
    print(f"Observation Shape: {obs.shape}")
    assert obs.shape == (4, 84, 84)
    env.close()