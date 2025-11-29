import gymnasium as gym
import numpy as np
from gymnasium import ObservationWrapper, Wrapper
from gymnasium.wrappers import FrameStackObservation
import cv2
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation,  MaxAndSkipObservation, ReshapeObservation
import ale_py
import matplotlib.pyplot as plt

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

def make_env(env_name, render=None):
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode=render)
    print("Standard Env.        :", env.observation_space.shape)

    env = MaxAndSkipObservation(env, skip=4)
    print("MaxAndSkipObservation:", env.observation_space.shape)

    env = ResizeObservation(env, (84, 84))
    print("ResizeObservation    :", env.observation_space.shape)

    env = GrayscaleObservation(env, keep_dim=True)
    print("GrayscaleObservation :", env.observation_space.shape)

    env = ImageToPyTorch(env)
    print("ImageToPyTorch       :", env.observation_space.shape)

    env = ReshapeObservation(env, (84, 84))
    print("ReshapeObservation   :", env.observation_space.shape)

    env = FrameStackObservation(env, stack_size=4)
    print("FrameStackObservation:", env.observation_space.shape)

    env = ScaledFloatFrame(env)
    print("ScaledFloatFrame     :", env.observation_space.shape)

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


def capture_and_save_pipeline(env_name="SkiingNoFrameskip-v4"):
    gym.register_envs(ale_py)
    
    #list to store (Stage Name, Observation Data, Shape)
    snapshots = []
    
    #1.base Env
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    snapshots.append(("Original Env", obs, obs.shape))

    #2.MaxAndSkipObservation
    env = MaxAndSkipObservation(env, skip=4)
    obs, _ = env.reset()
    snapshots.append(("MaxAndSkip", obs, obs.shape))

    #3.ResizeObservation
    env = ResizeObservation(env, (84, 84))
    obs, _ = env.reset()
    snapshots.append(("Resize (84x84)", obs, obs.shape))

    #4.GrayscaleObservation
    env = GrayscaleObservation(env, keep_dim=True)
    obs, _ = env.reset()
    snapshots.append(("Grayscale", obs, obs.shape))

    #5.ImageToPyTorch
    #swaps dimensions to (C, H, W)
    env = ImageToPyTorch(env)
    obs, _ = env.reset()
    snapshots.append(("ImageToPyTorch", obs, obs.shape))

    #6.ReshapeObservation
    #flattens (1, 84, 84) to (84, 84)
    env = ReshapeObservation(env, (84, 84))
    obs, _ = env.reset()
    snapshots.append(("Reshape", obs, obs.shape))

    #7.FrameStackObservation
    #stacks 4 frames. Result: (4, 84, 84)
    env = FrameStackObservation(env, stack_size=4)
    obs, _ = env.reset()
    #We step the environment to fill the stack with different frames for visualization
    action = env.action_space.sample()
    for _ in range(10):
        obs, _, _, _, _ = env.step(action)
    snapshots.append(("FrameStack (4)", obs, obs.shape))

    #8.ScaledFloatFrame
    #normalizes values to 0-1
    env = ScaledFloatFrame(env)
    obs, _ = env.reset() #Resetting clears the stack again, but that is fine for shape checking
    snapshots.append(("ScaledFloat", obs, obs.shape))

    #generate the plot
    save_plot(snapshots)

def save_plot(snapshots):
    num_snaps = len(snapshots)
    cols = 4
    rows = (num_snaps + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=(20, 10))
    axes = axes.flatten()

    for i, (name, obs, shape) in enumerate(snapshots):
        ax = axes[i]
        
        #prepare image for Matplotlib (needs H,W or H,W,C)
        img_display = obs
        
        #handle PyTorch format (C, H, W) or stacked (Stack, H, W)
        if len(obs.shape) == 3:
            if obs.shape[0] in [1, 4]: 
                #it is channel-first. We take the last frame/channel to display
                img_display = obs[-1]
            elif obs.shape[2] == 1:
                #it is (H, W, 1). Remove the last dim
                img_display = obs.squeeze()
        
        #plot
        if len(img_display.shape) == 2:
            ax.imshow(img_display, cmap='gray')
        else:
            ax.imshow(img_display)
            
        ax.set_title(f"{name}\n{shape}", fontsize=11)
        ax.axis('off')

    #hide empty subplots
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    #save the figure
    filename = "wrapper_steps.png"
    plt.savefig(filename)
    print(f"Successfully saved visualization to {filename}")
    plt.close()

capture_and_save_pipeline()