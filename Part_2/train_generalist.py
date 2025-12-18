import os
import numpy as np
import torch
import wandb
from datetime import datetime
import supersuit as ss
from pettingzoo.atari import pong_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecMonitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback, CheckpointCallback

# ------------------ DEVICE ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ------------------ WANDB CALLBACKS ------------------
class WandbCallback(BaseCallback):
    """
    Custom callback for logging training metrics to Wandb.
    Logs rewards, episode lengths, and training metrics.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        # Log episode rewards and lengths
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                self.episode_count += 1
                wandb.log({
                    "train/episode_reward": ep_info['r'],
                    "train/episode_length": ep_info['l'],
                    "train/episode_count": self.episode_count
                }, step=self.num_timesteps)
        
        return True

    def _on_rollout_end(self) -> None:
        """Log training metrics at the end of each rollout."""
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            wandb.log({
                "train/learning_rate": self.model.learning_rate,
            }, step=self.num_timesteps)


class WandbBestModelCallback(BaseCallback):
    """
    Callback to save best model to Wandb when a new best is found.
    Works in conjunction with EvalCallback.
    """
    def __init__(self, check_freq, save_path, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq
        self.save_path = save_path
        self.best_mean_reward = -float('inf')

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            best_model_path = os.path.join(self.save_path, "best_model.zip")
            if os.path.exists(best_model_path):
                try:
                    eval_file = os.path.join(self.save_path, "evaluations.npz")
                    if os.path.exists(eval_file):
                        import numpy as np
                        data = np.load(eval_file)
                        if len(data['results']) > 0:
                            current_best = data['results'][-1].mean()
                            if current_best > self.best_mean_reward:
                                self.best_mean_reward = current_best
                                wandb.save(best_model_path, base_path=os.path.dirname(self.save_path))
                                if self.verbose > 0:
                                    print(f"\n{'='*70}")
                                    print(f"🏆 NEW BEST MODEL!")
                                    print(f"Mean reward: {self.best_mean_reward:.2f}")
                                    print(f"Saved to Wandb and local: {best_model_path}")
                                    print(f"{'='*70}\n")
                                wandb.log({
                                    "eval/best_mean_reward": self.best_mean_reward
                                }, step=self.num_timesteps)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Error saving best model to Wandb: {e}")
        return True

# --------------- TERMINAL CALLBACK -------------
class ConsoleLoggerCallback(BaseCallback):
    """
    Prints training progress to the terminal.
    Calculates mean reward from the last 100 episodes.
    """
    def __init__(self, check_freq=1000, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.check_freq == 0:
            # Retrieve the episode buffer from the model (stores last 100 episodes)
            ep_info_buffer = self.model.ep_info_buffer
            if len(ep_info_buffer) > 0:
                mean_reward = np.mean([ep_info['r'] for ep_info in ep_info_buffer])
                mean_length = np.mean([ep_info['l'] for ep_info in ep_info_buffer])
                print(f"Step {self.num_timesteps: <8} | Mean Reward: {mean_reward: .2f} | Mean Length: {mean_length: .0f}")
            else:
                print(f"Step {self.num_timesteps: <8} | (No episodes finished yet)")
        return True
    

# ------------------ CONFIGURATION ------------------
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "PettingZoo/Pong-v3",
    "export_path": "./exports/pong_generalist/",
    "n_envs": 8,
    
    # Best Hyperparameters (from previous steps)
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
    
    # Evaluation
    "eval_freq": 10000,
    "n_eval_episodes": 10,
    "save_freq_epochs": 10,
}

# ------------------ ENVIRONMENT SETUP ------------------
def make_parallel_env(num_envs=8, seed=0):
    """
    Creates a vectorized PettingZoo environment compatible with SB3.
    Matches the tournament preprocessing EXACTLY.
    """
    # 1. Create Parallel PettingZoo Env
    env = pong_v3.parallel_env(num_players=2, render_mode=None)

    # 2. Apply SuperSuit Wrappers (The exact Tournament Pipeline)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0) # (4, 84, 84)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    
    # 3. Vectorize for SB3
    sb3_env = ss.pettingzoo_env_to_vec_env_v1(env)
    
    # 4. Concatenate multiple games
    num_games = max(1, num_envs // 2)
    sb3_env = ss.concat_vec_envs_v1(sb3_env, num_games, num_cpus=1, base_class='stable_baselines3')
    
    # 5. Add Monitor for SB3 logging
    sb3_env = VecMonitor(sb3_env)
    
    return sb3_env

# ------------------ TRAINING ------------------
def train_generalist_agent():
    print(f"\n{'='*70}")
    print(f"TRAINING GENERALIST AGENT (LEFT & RIGHT CAPABLE)")
    print(f"{'='*70}")
    
    # Create export directories
    os.makedirs(config["export_path"], exist_ok=True)
    best_model_path = os.path.join(config["export_path"], "best_model")
    checkpoint_path = os.path.join(config["export_path"], "checkpoints")
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(checkpoint_path, exist_ok=True)

    # Initialize Wandb
    run = wandb.init(
        project="Pong_Part2_Tournament",
        config=config,
        name=f"PPO_Generalist_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        sync_tensorboard=True,
        save_code=True,
        tags=["pong", "generalist", "self-play"]
    )

    # Create Environments
    print("Creating training environment...")
    env = make_parallel_env(num_envs=config["n_envs"], seed=42)
    
    print("Creating evaluation environment...")
    eval_env = make_parallel_env(num_envs=2, seed=123) # Minimum 2 for self-play logic

    # Initialize PPO Model
    print("Initializing PPO Model...")
    model = PPO(
        "CnnPolicy",
        env,
        verbose=1,
        device=DEVICE,
        tensorboard_log="./logs/pong_generalist",
        # Disable automatic normalization (SuperSuit handled it)
        policy_kwargs={"normalize_images": False}, 
        
        # Hyperparameters
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        learning_rate=config["learning_rate"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
    )

    # Calculate frequencies
    steps_per_update = config["n_steps"] * config["n_envs"]
    checkpoint_freq_timesteps = steps_per_update * config["save_freq_epochs"]

    # Setup Callbacks
    # Console Logger (Prints to terminal)
    console_callback = ConsoleLoggerCallback(check_freq=1000) # Print every 1000 steps

    # 1. Evaluation Callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=best_model_path,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        render=False,
        verbose=1
    )
    
    # 2. Checkpoint Callback
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq_timesteps,
        save_path=checkpoint_path,
        name_prefix="ppo_gen_epoch",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    # 3. Wandb Callback
    wandb_callback = WandbCallback(verbose=0)
    
    # 4. Best Model to Wandb
    wandb_best_callback = WandbBestModelCallback(
        check_freq=config["eval_freq"],
        save_path=best_model_path,
        verbose=1
    )

    callback_list = CallbackList([
        console_callback,
        wandb_callback,
        eval_callback,
        checkpoint_callback,
        wandb_best_callback
    ])

    print("Starting training... (Self-Play)")
    
    try:
        model.learn(
            total_timesteps=config["total_timesteps"], 
            callback=callback_list, 
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
    
    # Save Final Model
    final_path = os.path.join(config["export_path"], "ppo_generalist_final")
    model.save(final_path)
    print(f"\nFinal model saved to {final_path}.zip")
    
    # Upload to WandB
    wandb.save(f"{final_path}.zip", base_path=config["export_path"])
    
    env.close()
    eval_env.close()
    run.finish()
    
    return final_path

if __name__ == "__main__":
    train_generalist_agent()