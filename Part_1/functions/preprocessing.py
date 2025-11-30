import numpy as np
import gymnasium as gym
from gymnasium.wrappers import MaxAndSkipObservation, ResizeObservation, GrayscaleObservation, FrameStackObservation, ReshapeObservation
import ale_py
import matplotlib.pyplot as plt

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



def make_env(env_name, render = None):
    gym.register_envs(ale_py)
    env = gym.make(env_name, render_mode=render) # get the environment
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





def visualize_preprocessing(env_name="PongNoFrameskip-v4"):

    gym.register_envs(ale_py)

    snapshots = []

    # 0. Base env (with render)
    env = gym.make(env_name, render_mode="rgb_array")
    obs, _ = env.reset()
    snapshots.append(("Original (RGB)", obs, obs.shape))

    # 1. MaxAndSkip
    env = MaxAndSkipObservation(env, skip=4)
    obs, _ = env.reset()
    snapshots.append(("MaxAndSkip", obs, obs.shape))

    # 2. FireResetEnv (no visual change, only adds FIRE action on reset)
    env = FireResetEnv(env)
    obs, _ = env.reset()
    snapshots.append(("FireResetEnv (same image)", obs, obs.shape))

    # 3. Resize 84x84
    env = ResizeObservation(env, (84, 84))
    obs, _ = env.reset()
    snapshots.append(("Resize (84x84)", obs, obs.shape))

    # 4. Grayscale
    env = GrayscaleObservation(env, keep_dim=True)
    obs, _ = env.reset()
    snapshots.append(("Grayscale", obs, obs.shape))

    # 5. ImageToPyTorch (HWC → CHW)
    env = ImageToPyTorch(env)
    obs, _ = env.reset()
    snapshots.append(("ImageToPyTorch (C,H,W)", obs, obs.shape))

    # 6. ReshapeObservation (remove extra dim)
    env = ReshapeObservation(env, (84, 84))
    obs, _ = env.reset()
    snapshots.append(("Reshape (CHW→HW?)", obs, obs.shape))

    # 7. FrameStack
    env = FrameStackObservation(env, stack_size=4)
    obs, _ = env.reset()

    # Fill frames by repeating an action so stack shows motion
    action = env.action_space.sample()
    for _ in range(10):
        obs, _, _, _, _ = env.step(action)

    snapshots.append(("FrameStack (4)", obs, obs.shape))

    # 8. Scale to [0,1]
    env = ScaledFloatFrame(env)
    obs, _ = env.reset()
    snapshots.append(("ScaledFloatFrame", obs, obs.shape))

    save_plot(snapshots)


def save_plot(snapshots):
    cols = 4
    rows = (len(snapshots) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(22, 10))
    axes = axes.flatten()

    for i, (name, obs, shape) in enumerate(snapshots):
        ax = axes[i]

        img = obs

        # If CHW → show last channel or stack frame
        if len(img.shape) == 3:
            C, H, W = img.shape
            if C == 4:          # FrameStack → show last frame
                img = img[-1]
            elif C == 1:        # (1,H,W) grayscale
                img = img[0]

        # grayscale
        if len(img.shape) == 2:
            ax.imshow(img, cmap="gray")
        else:
            ax.imshow(img)

        ax.set_title(f"{name}\n{shape}")
        ax.axis("off")

    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig("env_preprocessing_visualized.png")
    print("Saved as env_preprocessing_visualized.png")
    plt.close()


visualize_preprocessing()
