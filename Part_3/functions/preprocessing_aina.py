import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper
from gymnasium.wrappers import (
    FrameStackObservation, 
    ResizeObservation, 
    GrayscaleObservation, 
    MaxAndSkipObservation,
    ReshapeObservation,
    TimeLimit
)
import ale_py


class CropObs(ObservationWrapper):
    """Crop observation to remove scoreboard and unnecessary parts."""
    def __init__(self, env, x_min=8, x_max=152, y_min=30, y_max=180):
        super().__init__(env)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        old_shape = env.observation_space.shape
        new_height = y_max - y_min
        new_width = x_max - x_min
        
        if len(old_shape) == 3:
            new_shape = (new_height, new_width, old_shape[2])
        else:
            new_shape = (new_height, new_width)
            
        self.observation_space = gym.spaces.Box(
            low=0, high=255, 
            shape=new_shape, 
            dtype=env.observation_space.dtype
        )

    def observation(self, obs):
        return obs[self.y_min:self.y_max, self.x_min:self.x_max]


class SkiingRewardShaping(gym.RewardWrapper):
    """
    Better reward shaping for Skiing.
    
    In Skiing, rewards are negative (time penalties and collision penalties).
    Raw rewards can be very large negative numbers (e.g., -8000+).
    
    We scale and clip to make learning more stable.
    """
    def __init__(self, env, scale=0.01, clip_min=-1.0, clip_max=1.0):
        super().__init__(env)
        self.scale = scale
        self.clip_min = clip_min
        self.clip_max = clip_max

    def reward(self, reward):
        # Scale the reward
        scaled_reward = reward * self.scale
        
        # Clip to prevent extreme values
        clipped_reward = np.clip(scaled_reward, self.clip_min, self.clip_max)
        
        return clipped_reward


class EpisodeInfoWrapper(gym.Wrapper):
    """Track episode statistics."""
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
        
        if terminated or truncated:
            info['episode'] = {
                'r': self.episode_reward,
                'l': self.episode_length
            }
        
        return obs, reward, terminated, truncated, info


def old_make_env(env_name="ALE/Skiing-v5", render=None, verbose=False):
    """
    Create and wrap Skiing environment with preprocessing pipeline.
    
    Pipeline:
    1. Create base environment
    2. MaxAndSkipObservation: Skip 4 frames, take max over last 2
    3. Reward shaping: Scale and clip rewards
    4. Crop: Remove scoreboard
    5. Resize: 84x84
    6. Grayscale: Convert to grayscale
    7. Reshape: Remove channel dimension
    8. FrameStack: Stack 4 frames for temporal information
    9. TimeLimit: Limit episode length
    10. EpisodeInfo: Track episode statistics
    """
    gym.register_envs(ale_py)
    
    # Base environment
    env = gym.make(env_name, render_mode=render)
    if verbose:
        print(f"{'Step':<25} {'Shape':<20} {'Description'}")
        print(f"{'-'*70}")
        print(f"{'Original Env':<25} {str(env.observation_space.shape):<20} Base Atari environment")
    
    # Frame skipping - take max over last 2 frames, skip 4 total
    env = MaxAndSkipObservation(env, skip=4)
    if verbose:
        print(f"{'MaxAndSkipObservation':<25} {str(env.observation_space.shape):<20} Skip 4 frames, max pooling")
    
    # Reward shaping
    # env = SkiingRewardShaping(env, scale=0.01, clip_min=-1.0, clip_max=1.0)
    # if verbose:
    #     print(f"{'SkiingRewardShaping':<25} {'':<20} Scale and clip rewards")
    
    # Crop to remove scoreboard and unnecessary parts
    env = CropObs(env, x_min=8, x_max=152, y_min=30, y_max=180)
    if verbose:
        print(f"{'CropObs':<25} {str(env.observation_space.shape):<20} Remove scoreboard")
    
    # Resize to 84x84
    env = ResizeObservation(env, (84, 84))
    if verbose:
        print(f"{'ResizeObservation':<25} {str(env.observation_space.shape):<20} Resize to 84x84")
    
    # Convert to grayscale
    env = GrayscaleObservation(env, keep_dim=True)
    if verbose:
        print(f"{'GrayscaleObservation':<25} {str(env.observation_space.shape):<20} Convert to grayscale")
    
    # Reshape to remove channel dimension for frame stacking
    env = ReshapeObservation(env, (84, 84))
    if verbose:
        print(f"{'ReshapeObservation':<25} {str(env.observation_space.shape):<20} Remove channel dim")
    
    # Stack 4 frames for temporal information
    env = FrameStackObservation(env, stack_size=4)
    if verbose:
        print(f"{'FrameStackObservation':<25} {str(env.observation_space.shape):<20} Stack 4 frames")
    
    # Limit episode length to prevent infinite episodes
    env = TimeLimit(env, max_episode_steps=25000)
    if verbose:
        print(f"{'TimeLimit':<25} {'':<20} Max 25000 steps/episode")
    
    # Add episode info tracking
    env = EpisodeInfoWrapper(env)
    if verbose:
        print(f"{'EpisodeInfoWrapper':<25} {'':<20} Track episode stats")
        print(f"{'-'*70}")
    
    return env

class SkiingSurvivalWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev_flags = 999
        self.last_x_pos = None
        self.frames_since_flag = 0  
        self.MAX_PATIENCE = 300     
        
        # --- pose definitions ---
        
        #slow poses - punish them
        self.FORBIDDEN_POSES = [0, 1, 2, 3, 4, 11, 12, 13, 14, 15]
        
        #going straight
        self.STRAIGHT_POSES = [7, 8]
        
        #balance between turning and going straight
        self.CARVING_POSES = [5, 6, 9, 10]

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.last_x_pos = None
        self.frames_since_flag = 0
        
        ram = self.env.unwrapped.ale.getRAM()
        if 107 < len(ram):
            self.prev_flags = int(ram[107])
            self.last_x_pos = int(ram[25])
        else:
            self.prev_flags = 999
            
        return obs, info

    def step(self, action):
        obs, native_reward, terminated, truncated, info = self.env.step(action)
        ram = self.env.unwrapped.ale.getRAM() #get ram
        
        current_flags = int(ram[107]) 
        player_x = int(ram[25])
        pose = int(ram[15])

        if self.last_x_pos is None: self.last_x_pos = player_x
        
        my_reward = 0.0
        self.frames_since_flag += 1

        #flags
        diff = self.prev_flags - current_flags
        if diff > 0 and diff < 20: #if takes a flag (with some margin)
            my_reward += 50.0  #flag reward
            self.frames_since_flag = 0 #reset patience
        self.prev_flags = current_flags

        #timeout penalty (for when it is horizontal)
        if self.frames_since_flag > self.MAX_PATIENCE:
            #punish it
            my_reward -= 20.0 
            truncated = True   
        
        #punish taking time
        my_reward -= 0.1

        #pose rewards/punishment
        
        if pose in self.FORBIDDEN_POSES:
            #we penalize going horizontal 
            my_reward -= 5.0 
        
        #reward going straight
        elif pose in self.STRAIGHT_POSES:
            my_reward += 0.1
        
        #reward carving a lot
        elif pose in self.CARVING_POSES:
            my_reward += 1.5

        #extra punishment
        if pose in [71, 72]: #when falling (collision)
            my_reward -= 10.0
            self.frames_since_flag = 0 

        #being close to the edge
        if player_x < 30 or player_x > 130:
            my_reward -= 5.0 

        self.last_x_pos = player_x

        return obs, my_reward, terminated, truncated, info
def make_env(env_name="ALE/Skiing-v5", render=None, verbose=False):
    """
    Create and wrap Skiing environment with preprocessing pipeline.
    
    Pipeline:
    1. Create base environment
    2. MaxAndSkipObservation: Skip 4 frames, take max over last 2
    3. Reward shaping: Scale and clip rewards
    4. Crop: Remove scoreboard
    5. Resize: 84x84
    6. Grayscale: Convert to grayscale
    7. Reshape: Remove channel dimension
    8. FrameStack: Stack 4 frames for temporal information
    9. TimeLimit: Limit episode length
    10. EpisodeInfo: Track episode statistics
    
    Args:
        env_name: Name of the Gymnasium environment
        render: Render mode (None, 'human', 'rgb_array')
        verbose: Print detailed information about the pipeline
        
    Returns:
        Wrapped Gymnasium environment
    """
    gym.register_envs(ale_py)
    
    # Base environment
    env = gym.make(env_name, render_mode=render)
    if verbose:
        print(f"\n{'='*70}")
        print(f"ENVIRONMENT PREPROCESSING PIPELINE")
        print(f"{'='*70}")
        print(f"{'Step':<25} {'Shape':<20} {'Description'}")
        print(f"{'-'*70}")
        print(f"{'Original Env':<25} {str(env.observation_space.shape):<20} Base Atari environment")
    
    env = SkiingSurvivalWrapper(env)

    # Frame skipping - take max over last 2 frames, skip 4 total
    env = MaxAndSkipObservation(env, skip=4)
    if verbose:
        print(f"{'MaxAndSkipObservation':<25} {str(env.observation_space.shape):<20} Skip 4 frames, max pooling")
    
    # Reward shaping - LESS aggressive scaling
    # env = SkiingRewardShaping(env, scale=0.1, clip_min=-10.0, clip_max=10.0)
    # env = SkiingRewardShaping(env, scale=0.05, clip_min=-100.0, clip_max=100.0)
    # if verbose:
    #     print(f"{'SkiingRewardShaping':<25} {'':<20} Scale rewards by 0.1, clip to [-10, 10]")
    
    # Crop to remove scoreboard and unnecessary parts
    env = CropObs(env, x_min=8, x_max=152, y_min=30, y_max=180)
    if verbose:
        print(f"{'CropObs':<25} {str(env.observation_space.shape):<20} Remove scoreboard")
    
    # Resize to 84x84
    env = ResizeObservation(env, (84, 84))
    if verbose:
        print(f"{'ResizeObservation':<25} {str(env.observation_space.shape):<20} Resize to 84x84")
    
    # Convert to grayscale
    env = GrayscaleObservation(env, keep_dim=True)
    if verbose:
        print(f"{'GrayscaleObservation':<25} {str(env.observation_space.shape):<20} Convert to grayscale")
    
    # Reshape to remove channel dimension for frame stacking
    env = ReshapeObservation(env, (84, 84))
    if verbose:
        print(f"{'ReshapeObservation':<25} {str(env.observation_space.shape):<20} Remove channel dim")
    
    # Stack 4 frames for temporal information
    env = FrameStackObservation(env, stack_size=4)
    if verbose:
        print(f"{'FrameStackObservation':<25} {str(env.observation_space.shape):<20} Stack 4 frames")
    
    # Limit episode length to prevent infinite episodes
    env = TimeLimit(env, max_episode_steps=25000)
    if verbose:
        print(f"{'TimeLimit':<25} {'':<20} Max 25000 steps/episode")
    
    # Add episode info tracking
    env = EpisodeInfoWrapper(env)
    if verbose:
        print(f"{'EpisodeInfoWrapper':<25} {'':<20} Track episode stats")
        print(f"{'-'*70}")
        print(f"{'='*70}\n")
    
    return env


def test_environment():
    """Test the environment setup."""
    print("\n" + "="*70)
    print("TESTING ENVIRONMENT SETUP")
    print("="*70 + "\n")
    
    env = make_env("ALE/Skiing-v5", verbose=True, render="human")
    
    print("\n" + "="*70)
    print("RUNNING TEST EPISODE")
    print("="*70 + "\n")
    
    obs, info = env.reset()
    total_reward = 0
    steps = 0
    
    print(f"Observation shape: {obs.shape}")
    print(f"Observation dtype: {obs.dtype}")
    print(f"Observation range: [{obs.min():.2f}, {obs.max():.2f}]")
    print(f"\nAction space: {env.action_space}")
    print(f"Number of actions: {env.action_space.n}\n")
    
    # Run for a few steps
    for i in range(500):
        action = env.action_space.sample()
        obs, reward, terminated, truncated, info = env.step(action)
        print(f"Action: {action} | Reward: {reward:.2f}")
        total_reward += reward
        steps += 1
        
        if terminated or truncated:
            print(f"Episode ended at step {steps}")
            print(f"Total reward: {total_reward:.2f}")
            if 'episode' in info:
                print(f"Episode info: {info['episode']}")
            break
    
    if steps == 100:
        print(f"Ran {steps} steps")
        print(f"Total reward so far: {total_reward:.2f}")
    
    env.close()
    print("\n" + "="*70)
    print("TEST COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    test_environment()