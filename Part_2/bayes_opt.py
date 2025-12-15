# pong_bayes_opt.py
import os
import json
import argparse
from datetime import datetime
import numpy as np
import optuna
import torch
import wandb
from wandb.integration.sb3 import WandbCallback
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
    "export_path": "/datafast/105-1/Datasets/INTERNS/anavarror/paradigms/exports/bayes_opt/",
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
                if self.episode_count % 10 == 0:
                    print(f"Trial Episode {self.episode_count}: Reward {ep.get('r'):.1f}, Length {ep.get('l')}")
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

    # -------------------------------------------------------------
    # OPTUNA HYPERPARAMETER SEARCH SPACE
    # (Updated to match recommended ranges)
    # -------------------------------------------------------------
    cfg["n_steps"] = trial.suggest_categorical("n_steps", [128, 256, 512])
    cfg["gamma"] = trial.suggest_categorical("gamma", [0.99, 0.995, 0.999])
    cfg["learning_rate"] = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
    cfg["ent_coef"] = trial.suggest_float("ent_coef", 0.00001, 0.01, log=True)
    cfg["clip_range"] = trial.suggest_categorical("clip_range", [0.1, 0.2])
    cfg["n_epochs"] = trial.suggest_categorical("n_epochs", [4, 10])
    cfg["gae_lambda"] = trial.suggest_categorical("gae_lambda", [0.95, 0.98])
    cfg["max_grad_norm"] = trial.suggest_categorical("max_grad_norm", [0.5, 1.0])
    cfg["vf_coef"] = trial.suggest_float("vf_coef", 0.5, 1.0)
    cfg["batch_size"] = trial.suggest_categorical("batch_size", [64, 128, 256])

    # VALIDATION: Ensure batch_size divides evenly into buffer_size
    # PPO requires (n_steps * n_envs) % batch_size == 0
    buffer_size = cfg["n_steps"] * cfg["n_envs"]
    if buffer_size % cfg["batch_size"] != 0:
        raise optuna.exceptions.TrialPruned()
    
    trial_start_time = datetime.now()
    print(f"\n{'#'*80}")
    print(f"TRIAL {trial.number} STARTING at {trial_start_time.strftime('%H:%M:%S')}")
    print(f"{'#'*80}")
    print("HYPERPARAMETERS:")
    for key, value in cfg.items():
        if key not in config: continue # Only print optimization params if preferred, or all
        print(f"  {key:<20}: {value}")
    print("-" * 80 + "\n")

    # 1. INIT WANDB RUN
    # We use reinit=True to allow multiple runs in the same script
    # grouping by "optuna_search" lets you filter them easily in the dashboard
    run = wandb.init(
        project="Pong_Part2_Tournament", 
        name=f"trial_{trial.number}", 
        group="optuna_search",
        config=cfg,
        reinit=True, 
        sync_tensorboard=True,  # Auto-upload SB3 metrics
        save_code=True,
    )

    try:
        # Create training environment
        train_env = make_vec_env(cfg["env_name"], cfg["n_envs"], seed=trial.number)
        model = build_model(train_env, cfg, verbose=0)

        logging_cb = CustomLoggingCallback()  # Logging callback for per-episode metrics
        wandb_cb = WandbCallback(
            gradient_save_freq=0,
            model_save_path=None,
            verbose=0
        )

        # Evaluation environment for pruning
        prune_eval_env = make_env(cfg["env_name"])
        prune_eval_env = Monitor(prune_eval_env)

        pruning_cb = OptunaPruningCallback(
            trial=trial,
            eval_env=prune_eval_env,
            eval_freq=50_000,
            n_eval_episodes=20, # Reduced episodes for speed during pruning checks
        )

        # Train the model
        model.learn(
            total_timesteps=5_000_000, # Note: Trials will be pruned early if bad
            callback=CallbackList([logging_cb, pruning_cb, wandb_cb]),
            progress_bar=True
        )

        # Final evaluation
        eval_env = make_env(cfg["env_name"])
        eval_env = Monitor(eval_env)

        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=100)

        # Log final result explicitly
        wandb.log({"final_eval_reward": mean_reward})

        # Save JSON with episode logs
        save_trial_json(trial.number, cfg, logging_cb.episode_logs)

        print(f"\n--> Trial {trial.number} Finished. Mean Reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    except Exception as e:
        # If trial crashes or is pruned, mark as failed in wandb
        print(f"Trial failed: {e}")
        wandb.finish(exit_code=1)
        raise e

    finally:
        trial_end_time = datetime.now()
        duration = trial_end_time - trial_start_time
        print(f"Trial {trial.number} ENDED at {trial_end_time.strftime('%H:%M:%S')} (Duration: {duration})")
        print(f"{'#'*80}\n")
        
        # Log duration to wandb summary before finishing
        wandb.run.summary["trial_duration_seconds"] = duration.total_seconds()

        run.finish()

        # Clean up
        try:
            eval_env.close()
        except: pass
        try:
            train_env.close()
        except: pass
        try:
            prune_eval_env.close()
        except: pass

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
        overall_start = datetime.now()
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION STUDY STARTED AT: {overall_start.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"{'='*80}\n")

        storage_url = "/datafast/105-1/Datasets/INTERNS/anavarror/paradigms/bayes_opt_aina/sqlite:///pong_optuna.db"
        study_name = "/datafast/105-1/Datasets/INTERNS/anavarror/paradigms/bayes_opt_aina/pong_ppo_optimization"

        # MedianPruner is efficient for stopping bad trials early
        study = optuna.create_study(
            direction="maximize", 
            pruner=optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=100_000)
        )
        study.optimize(objective, n_trials=args.optimize)
        save_all_trials_txt(study)

        overall_end = datetime.now()
        total_duration = overall_end - overall_start
        print(f"\n{'='*80}")
        print(f"OPTIMIZATION STUDY FINISHED AT: {overall_end.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"TOTAL DURATION: {total_duration}")
        print(f"BEST PARAMS: {study.best_params}")
        print(f"{'='*80}\n")