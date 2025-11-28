import os
import numpy as np
import wandb
from pettingzoo.atari import pong_v3
import supersuit as ss

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback
from stable_baselines3.common.monitor import Monitor


# CALLBACK FOR W&B LOGGING
class CustomLoggingCallback(BaseCallback):
    def __init__(self, agent_name="agent", avg_window=20, verbose=0):
        super().__init__(verbose)
        self.avg_window = avg_window
        self.agent_name = agent_name
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info is not None:
                ep_rew = ep_info["r"]
                ep_len = ep_info["l"]
                self.episode_rewards.append(ep_rew)
                self.episode_lengths.append(ep_len)
                wandb.log({
                    f"{self.agent_name}/episode_reward": ep_rew,
                    f"{self.agent_name}/episode_length": ep_len,
                })
        return True

    def _on_rollout_end(self):
        if len(self.episode_rewards) > 0:
            avg_rew = np.mean(self.episode_rewards[-self.avg_window:])
            avg_len = np.mean(self.episode_lengths[-self.avg_window:])
            wandb.log({
                f"{self.agent_name}/avg_episode_reward": avg_rew,
                f"{self.agent_name}/avg_episode_length": avg_len,
            })


# ENVIRONMENT CREATION
def make_pong_env():
    env = pong_v3.env(obs_type="grayscale_image", frameskip=4, render_mode=None)
    env = ss.resize_v0(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, stack_size=4)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=255)
    return env

def make_vec_env():
    def _init():
        env = make_pong_env()
        env = Monitor(env)
        return env
    return DummyVecEnv([_init])


# TRAIN LEFT PADDLE AGAINST RIGHT PADDLE
def train_left_against_right(
    right_model_path: str,
    total_timesteps=500_000,
    avg_window=20,
    save_every_episodes=50,
    gamma=0.99,
    batch_size=64,
    learning_rate=2.5e-4
):
    """
    Train left paddle agent against pre-trained right paddle agent.
    Hyperparameters gamma, batch_size, and learning_rate are exposed for grid search.
    """

    # Load pre-trained right paddle (evaluation mode)
    right_agent = PPO.load(right_model_path, device="cuda")
    right_agent.policy.eval()

    # Initialize W&B run
    run = wandb.init(
        project="multiagent_pong",
        config={
            "total_timesteps": total_timesteps,
            "avg_window": avg_window,
            "gamma": gamma,
            "batch_size": batch_size,
            "learning_rate": learning_rate
        },
        sync_tensorboard=True,
        save_code=True,
    )

    # Environment for left agent
    env = make_vec_env()

    # Logging and checkpoint callbacks
    logging_callback = CustomLoggingCallback(agent_name="left_agent", avg_window=avg_window)
    approx_steps_per_episode = 5000
    checkpoint_callback = CheckpointCallback(
        save_freq=save_every_episodes * approx_steps_per_episode,
        save_path="./exports_left/",
        name_prefix="left_agent_checkpoint"
    )

    # Initialize left paddle PPO with hyperparameters for grid search
    agent_left = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device="cuda",
        gamma=gamma,
        batch_size=batch_size,
        learning_rate=learning_rate,
        tensorboard_log="./logs_left/"
    )

    # Train
    agent_left.learn(
        total_timesteps=total_timesteps,
        callback=[logging_callback, checkpoint_callback]
    )

    # Save final model
    agent_left.save("./exports_left/left_agent_final")
    print("Left agent trained against fixed right agent. Model saved.")

    run.finish()


# RUN TRAINING
if __name__ == "__main__":
    right_model_path = "./exports/right_agent_final.zip"
    train_left_against_right(
        right_model_path,
        total_timesteps=500_000,
        avg_window=20,
        save_every_episodes=50,
        gamma=0.99,
        batch_size=64,
        learning_rate=2.5e-4
    )
