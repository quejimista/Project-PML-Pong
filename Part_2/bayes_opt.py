# pong_bayes_opt.py
import os
import json
import argparse
from datetime import datetime
import numpy as np
import optuna
import torch

import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import FrameStackObservation, ResizeObservation
import ale_py

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback, CallbackList
from stable_baselines3.common.evaluation import evaluate_policy



# Preprocessing wrappers

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
            low=0,
            high=255,
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
            info["episode"] = {
                "r": self.episode_reward,
                "l": self.episode_length,
            }
        return obs, reward, terminated, truncated, info


def make_env(env_name="ALE/Pong-v5", render=None, verbose=False):
    """
    Create and wrap Pong environment.
    Final observation space: Box(0.0, 1.0, (4, 84, 84), float32)
    """
    gym.register_envs(ale_py)  # Base environment
    env = gym.make(env_name, render_mode=render)

    if verbose:
        print(f"Original: {env.observation_space.shape}")

    env = ColorReduction(env)  # Color reduction (to 2D)
    env = ResizeObservation(env, (84, 84))  # Resize to 84x84

    #Frame stack (4) -> Result will be (4, 84, 84) because input is (84, 84)
    # Gymnasium FrameStack stacks on the first dimension.
    env = FrameStackObservation(env, stack_size=4)
    env = DtypeWrapper(env, np.float32)  # Convert dtype to float32
    env = NormalizeObservation(env, env_min=0, env_max=1)  # Normalize observations
    env = EpisodeInfoWrapper(env)  # Add episode info tracking

    if verbose:
        print(f"Final Space: {env.observation_space}")

    return env



# Device selection and default training config

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "ALE/Pong-v5",
    "export_path": "./exports/pong/",
    "n_envs": 8,
    "n_steps": 128,
    "batch_size": 256,
    "n_epochs": 4,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.1,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "learning_rate": 2.5e-4,
    "eval_freq": 10000,
    "n_eval_episodes": 10,
    "save_freq_epochs": 10,
}



# Vectorized env factory

def make_vec_env(env_name, n_envs, seed=0):
    """
    Create a vectorized environment for parallel training.
    Uses SubprocVecEnv for multiprocessing to speed up training.
    """
    def _make_env(rank):
        def _init():
            env = make_env(env_name)
            env = Monitor(env)  # Track episode rewards for SB3
            env.reset(seed=seed + rank)
            return env
        return _init
    
    # Create subprocess vectorized environment
    vec_env = SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)  # Monitor all subprocesses
    return vec_env



# Model factory

def build_model(env, cfg, verbose=0):
    """
    Build a PPO agent with given environment and hyperparameters.
    """
    return PPO(
        cfg["policy_type"],
        env,
        verbose=verbose,
        device=DEVICE,
        tensorboard_log=None,
        policy_kwargs={"normalize_images": False},

        n_steps=cfg["n_steps"],
        batch_size=cfg["batch_size"],
        n_epochs=cfg["n_epochs"],
        gamma=cfg["gamma"],
        gae_lambda=cfg["gae_lambda"],
        clip_range=cfg["clip_range"],
        ent_coef=cfg["ent_coef"],
        vf_coef=cfg["vf_coef"],
        max_grad_norm=cfg["max_grad_norm"],
        learning_rate=cfg["learning_rate"],
    )



# Custom logging callback

class CustomLoggingCallback(BaseCallback):
    """
    Log per-episode metrics during training:
    - episode reward
    - episode length
    - policy/value/entropy losses (if available)
    Store results in a list for later saving.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_logs = []
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep = info.get("episode")
            if ep is not None:
                self.episode_count += 1
                self.episode_logs.append({
                    "episode": self.episode_count,
                    "reward": float(ep.get("r", 0.0)),
                    "length": float(ep.get("l", 0)),
                    "policy_loss": float(self.locals.get("policy_loss", float("nan"))) 
                                    if "policy_loss" in self.locals else None,
                    "value_loss": float(self.locals.get("value_loss", float("nan"))) 
                                    if "value_loss" in self.locals else None,
                    "entropy_loss": float(self.locals.get("entropy_loss", float("nan"))) 
                                    if "entropy_loss" in self.locals else None
                })
        return True


# Save trial logs to JSON

def save_trial_json(trial_id, cfg, logs, output_dir="trial_logs"):
    """
    Save all episode logs of a trial to a JSON file.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(output_dir, f"trial_{trial_id}_{ts}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "trial": trial_id,
                "config": cfg,
                "episodes": logs,
            },
            f,
            indent=2,
        )
    print(f"[INFO] Saved trial JSON: {path}")


# Save all trials summary to text file

def save_all_trials_txt(study, output_path="all_hyperparams.txt"):
    """
    Save summary of all Optuna trials in a human-readable text file.
    Logs --> trial number, mean reward, hyperparameters, and trial state.
    """
    with open(output_path, "w") as f:
        for t in study.trials:
            f.write(f"Trial {t.number}\n")
            f.write(f"Mean eval reward: {t.value}\n")
            f.write(f"Params: {t.params}\n")
            f.write(f"State: {t.state}\n\n")
    print(f"[INFO] Saved all trials to {output_path}")


# Optuna pruning callback

class OptunaPruningCallback(BaseCallback):
    """
    Evaluate model periodically and reports intermediate rewards to Optuna.
    Allow early stopping (pruning) of underperforming trials.
    """
    def __init__(self, trial, eval_env, eval_freq, n_eval_episodes, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.last_eval_step = 0

    def _on_step(self) -> bool:
        # Only evaluate every eval_freq steps
        if self.num_timesteps - self.last_eval_step < self.eval_freq:
            return True

        self.last_eval_step = self.num_timesteps
        
        # Evaluate the policy
        mean_reward, _ = evaluate_policy(
            self.model,
            self.eval_env,
            n_eval_episodes=self.n_eval_episodes,
            deterministic=True,
        )

        # Report intermediate result to Optuna
        self.trial.report(mean_reward, step=self.num_timesteps)

        # Ask Optuna if this trial should be pruned
        if self.trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        return True



# Optuna objective function

def objective(trial):
    """
    Define the hyperparameter search space and runs a single training/evaluation trial.
    Return mean evaluation reward as the objective value.
    """
    cfg = config.copy()

    # Define Optuna search space
    cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    cfg["gamma"] = trial.suggest_float("gamma", 0.95, 0.999)
    cfg["gae_lambda"] = trial.suggest_float("gae_lambda", 0.9, 0.99)
    cfg["clip_range"] = trial.suggest_float("clip_range", 0.05, 0.2)
    cfg["ent_coef"] = trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True)

    cfg["n_steps"] = trial.suggest_categorical("n_steps", [64, 128, 256])
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256, 512])
    cfg["n_epochs"] = trial.suggest_int("n_epochs", 2, 6)

    # Ensure batch_size is valid
    if cfg["batch_size"] > cfg["n_steps"] * cfg["n_envs"]:
        raise optuna.exceptions.TrialPruned()

    # Create training environment
    train_env = make_vec_env(cfg["env_name"], cfg["n_envs"], seed=trial.number)
    model = build_model(train_env, cfg, verbose=0)

    logging_cb = CustomLoggingCallback()  # Logging callback for per-episode metrics

    # Evaluation environment for pruning
    prune_eval_env = make_env(cfg["env_name"])
    prune_eval_env = Monitor(prune_eval_env)

    pruning_cb = OptunaPruningCallback(
        trial=trial,
        eval_env=prune_eval_env,
        eval_freq=50_000,
        n_eval_episodes=50,
    )

    # Train the model
    model.learn(
        total_timesteps=3_000_000,
        callback=CallbackList([logging_cb, pruning_cb]),
    )

    # Final evaluation
    eval_env = make_env(cfg["env_name"])
    eval_env = Monitor(eval_env)

    mean_reward, _ = evaluate_policy(model, eval_env, n_eval_episodes=100)

    # Save JSON with episode logs
    save_trial_json(trial.number, cfg, logging_cb.episode_logs)

    # Clean up
    eval_env.close()
    train_env.close()
    prune_eval_env.close()

    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass
    
    # Return mean reward as objective
    return float(mean_reward)


# Entrypoint

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimize", type=int, default=0)
    args = parser.parse_args()


    if args.optimize > 0:
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=args.optimize)
        save_all_trials_txt(study)
