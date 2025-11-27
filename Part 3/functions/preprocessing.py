import numpy as np
import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation
import ale_py


class RestrictActions(gym.ActionWrapper):
    #Restrict action space to only cardinal directions (delete diagonal moves)"""
    def __init__(self, env, allowed_actions=[0, 1, 2, 3, 4]):
        """
        takes a Gymnasium environment (in our case Ms. Pac-Man) and the original actions
                Default [0,1,2,3,4] = ['NOOP', 'UP', 'RIGHT', 'LEFT', 'DOWN']
        """
        super().__init__(env)
        self.allowed_actions = allowed_actions
        self.action_space = gym.spaces.Discrete(len(allowed_actions))
        
        #store original action meanings for reference
        original_meanings = env.unwrapped.get_action_meanings()
        self.action_meanings = [original_meanings[i] for i in allowed_actions]
    
    def action(self, act):
        """map restricted action to original action space"""
        return self.allowed_actions[act]
    
    def get_action_meanings(self):
        #Return meanings of restricted actions
        return self.action_meanings

class ScaledFloatFrame(gym.ObservationWrapper):
    #set pixels to range 0-1
    def __init__(self, env):
        super().__init__(env)
        self.observation_space = gym.spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=env.observation_space.shape, 
            dtype=np.float32
        )
    
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0


class ClipRewardEnv(gym.RewardWrapper):
    """Clip rewards to {-1, 0, 1} for stability in training"""
    def reward(self, reward):
        return np.sign(reward)


def make_mspacman_env(render_mode=None, clip_rewards=True, restrict_actions=True):
    #create Ms. Pac-Man environment with standard Atari preprocessing.
    #returns the Wrapped environment with observations of shape (4, 84, 84)

    gym.register_envs(ale_py)
    env = gym.make("ALE/MsPacman-v5", render_mode=render_mode)
    
    print(f"Original environment: {env.observation_space.shape}")
    print(f"Original actions    : {env.unwrapped.get_action_meanings()}")
    
    #restrict to cardinal directions only (optional)
    if restrict_actions:
        env = RestrictActions(env, allowed_actions=[0, 1, 2, 3, 4]) # 0 = NOOP, 1 = UP, 2 = RIGHT, 3 = LEFT, 4 = DOWN
        print(f"Restricted actions  : {env.get_action_meanings()}")
    
    #skip frames
    env = MaxAndSkipObservation(env, skip=4)
    print(f"After MaxAndSkip    : {env.observation_space.shape}")
    
    #resize to 84x84
    env = ResizeObservation(env, (84, 84))
    print(f"After Resize        : {env.observation_space.shape}")
    
    #convert to grayscale (no extra dimension)
    env = GrayscaleObservation(env, keep_dim=False)
    print(f"After Grayscale     : {env.observation_space.shape}")
    
    #normalize to [0, 1]
    env = ScaledFloatFrame(env)
    print(f"After Scaling       : {env.observation_space.shape}")
    
    #stack 4 frames for temporal information
    env = FrameStackObservation(env, stack_size=4)
    print(f"After FrameStack    : {env.observation_space.shape}")


    #optional parameter: Clip rewards for stability, to {-1, 0, 1}
    if clip_rewards:
        env = ClipRewardEnv(env)
        print("Reward clipping enabled")
    
    return env


#example usage
#create environment
env = make_mspacman_env(render_mode="human")

#test the environment
obs, info = env.reset()
print(f"Observation dtype: {obs.dtype}")
print(f"Observation range: [{obs.min():.3f}, {obs.max():.3f}]")

#run a few steps
total_reward = 0
for _ in range(200):
    action = env.action_space.sample()
    obs, reward, terminated, truncated, info = env.step(action)
    total_reward += reward
    
    if terminated or truncated:
        print(f"\nEpisode finished! Total reward: {total_reward}")
        obs, info = env.reset()
        total_reward = 0

env.close()