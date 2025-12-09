import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.wrappers import FrameStackObservation
import cv2
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation,  MaxAndSkipObservation, ReshapeObservation
import ale_py
import matplotlib.pyplot as plt
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common.monitor import Monitor
class GrayScaleObs(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        obs_shape = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(obs_shape[0], obs_shape[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        gray = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        return gray[..., None] #np trick to add a new dimension



class ResizeObs(ObservationWrapper):
    def __init__(self, env, shape=(84, 84)):
        super().__init__(env)
        self.shape = shape
        self.observation_space = gym.spaces.Box(
            low=0, high=255,
            shape=(shape[0], shape[1], 1),
            dtype=np.uint8
        )

    def observation(self, obs):
        frame = cv2.resize(obs, self.shape, interpolation=cv2.INTER_AREA)
        return frame

class NormalizeObs(ObservationWrapper):
    def observation(self, obs):
        return obs.astype(np.float32) / 255.0

class ImageToPyTorch(gym.ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)
        old_shape = self.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(old_shape[-1], old_shape[0], old_shape[1]), dtype=np.float32) 
        #change the order of the matrix (frame). For using torch we need to swap the dimensions

    def observation(self, observation):
        return np.moveaxis(observation, 2, 0)

class ScaledFloatFrame(gym.ObservationWrapper):
    def observation(self, obs):
        return np.array(obs).astype(np.float32) / 255.0
    # change the range of values to [0,1]


#the idea of this wrapper is to reduce the noise of the observation. Ex leaderboard seconds and all that stuff,
#because it doesnt matter for the action to take
class CropObs(ObservationWrapper):
    def __init__(self, env, x_min=8, x_max=152, y_min=30, y_max=180):
        super().__init__(env)
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        
        #comput shape
        old_shape = env.observation_space.shape
        new_height = y_max - y_min
        new_width = x_max - x_min
        
        #keep channels if they existed
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
        #numpy slice
        #should work for (H, W) and (H, W, C)
        return obs[self.y_min:self.y_max, self.x_min:self.x_max]



#in our opinion better than clipping because that would make all negative rewards the same
#ex passing seconds the same as missing a gate
class SkiingRewardScaler(gym.RewardWrapper):
    def __init__(self, env, scale=0.01):
        super().__init__(env)
        self.scale = scale

    def reward(self, reward):
        #reduce reward by dividing (ex: -8515 -> -85.15)
        return reward * self.scale

class FrameSkip(Wrapper):
    def __init__(self, env, skip=4):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0
        done = truncated = False
        obs = None

        for _ in range(self.skip):
            obs, reward, done, truncated, info = self.env.step(action)
            total_reward += reward
            if done or truncated:
                break
        return obs, total_reward, done, truncated, info

def make_env(env_name = "ALE/Skiing-v5", render=None, verbose = False):
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode=render)
    if verbose:
        print("Standard Env.        :", env.observation_space.shape)

    env = MaxAndSkipObservation(env, skip=4)
    if verbose:
        print("MaxAndSkipObservation:", env.observation_space.shape)

    env = SkiingRewardScaler(env) 
    if verbose:
        print("Reward Scaled        : (Reward * scale factor)")

    env = CropObs(env, x_min=8, x_max=152, y_min=30, y_max=180)
    if verbose:
        print("CropObs              :", env.observation_space.shape)

    env = ResizeObservation(env, (84, 84))
    if verbose:
        print("ResizeObservation    :", env.observation_space.shape)

    env = GrayscaleObservation(env, keep_dim=True)
    if verbose:
        print("GrayscaleObservation :", env.observation_space.shape)

    # env = ImageToPyTorch(env)
    # print("ImageToPyTorch       :", env.observation_space.shape)

    env = ReshapeObservation(env, (84, 84))
    if verbose:
        print("ReshapeObservation   :", env.observation_space.shape)

    env = FrameStackObservation(env, stack_size=4)
    if verbose:
        print("FrameStackObservation:", env.observation_space.shape)

    # env = ScaledFloatFrame(env)
    # print("ScaledFloatFrame     :", env.observation_space.shape)
    env = TimeLimit(env, max_episode_steps=25000)
    env = Monitor(env)

    return env

# env = gym.make("ALE/Skiing-v5")
# obs, info = env.reset()
# total_reward = 0

# done = False
# while not done:
#     obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
#     total_reward += reward
#     done = terminated or truncated

# print("Total reward:", total_reward)


def capture_and_save_pipeline(env_name="ALE/Skiing-v5"):
    gym.register_envs(ale_py)
    
    snapshots = []
    
    # 1. base Env
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    snapshots.append(("Original Env", obs, obs.shape))

    # 2. MaxAndSkip
    env = MaxAndSkipObservation(env, skip=4)
    obs, _ = env.reset()
    snapshots.append(("MaxAndSkip", obs, obs.shape))

    # 3. Reward Scaler (no visual change to the image)
    env = SkiingRewardScaler(env, scale=0.05)
    
    # 4. CropObs (remove timer)
    env = CropObs(env, x_min=8, x_max=152, y_min=30, y_max=180)
    obs, _ = env.reset()
    snapshots.append(("CropObs (No Timer)", obs, obs.shape))

    # 5. ResizeObservation
    #resize the cropped img
    env = ResizeObservation(env, (84, 84))
    obs, _ = env.reset()
    snapshots.append(("Resize (84x84)", obs, obs.shape))

    # 6. GrayscaleObservation
    env = GrayscaleObservation(env, keep_dim=True)
    obs, _ = env.reset()
    snapshots.append(("Grayscale", obs, obs.shape))

    # 7. ImageToPyTorch
    env = ImageToPyTorch(env)
    obs, _ = env.reset()
    snapshots.append(("ImageToPyTorch", obs, obs.shape))

    # 8. ReshapeObservation
    env = ReshapeObservation(env, (84, 84))
    obs, _ = env.reset()
    snapshots.append(("Reshape", obs, obs.shape))

    # 9. FrameStackObservation
    env = FrameStackObservation(env, stack_size=4)
    obs, _ = env.reset()
    
    #fill stack a bit to see movement movement
    action = env.action_space.sample()
    for _ in range(12):
        obs, _, _, _, _ = env.step(action)
    snapshots.append(("FrameStack (4)", obs, obs.shape))

    # 10. ScaledFloatFrame
    env = ScaledFloatFrame(env)
    obs, _ = env.reset()
    snapshots.append(("ScaledFloat", obs, obs.shape))

    save_plot(snapshots)

def save_plot(snapshots):
    num_snaps = len(snapshots)
    cols = 4
    rows = (num_snaps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for i, (name, obs, shape) in enumerate(snapshots):
        ax = axes[i]
        
        img_display = obs
        
        #manage tensors (C, H, W) or Stacks
        if len(obs.shape) == 3:
            if obs.shape[0] in [1, 4] and obs.shape[0] < obs.shape[1]: 
                #its Channel-First -> take last frame
                img_display = obs[-1]
            elif obs.shape[2] == 1:
                #(H, W, 1) -> remove extra dim
                img_display = obs.squeeze()
        
        if len(img_display.shape) == 2:
            ax.imshow(img_display, cmap='gray')
        else:
            ax.imshow(img_display)
            
        ax.set_title(f"{name}\n{shape}", fontsize=10)
        ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    filename = "wrapper_steps_final.png"
    plt.savefig(filename)
    print(f"Successfully saved visualization to {filename}")
    plt.close()


if __name__ == "__main__":
    # capture_and_save_pipeline()

    #simulate a game to see how it moves
    env = make_env("ALE/Skiing-v5", render='human')
    obs, _ = env.reset()
    total_reward = 0

    print("Starting rewards calculation...")


    steps = 0
    while True:
        obs, reward, terminated, truncated, info = env.step(env.action_space.sample())
        steps += 1
        total_reward += reward

        if terminated or truncated:
            print("\n>>> Episode ended")
            print("terminated =", terminated)
            print("truncated =", truncated)
            print("steps =", steps)
            print("total_reward =", total_reward)
            break