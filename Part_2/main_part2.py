import wandb
import torch
import os
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback, CheckpointCallback
from stable_baselines3.common.evaluation import evaluate_policy
from preprocessing_pong import make_env


# ------------------ DEVICE ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ------------------ Wandb callbacks ------------------
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


# ------------------ CONFIG ------------------
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5_000_000,  # 5M timesteps for Pong
    "env_name": "ALE/Pong-v5",
    "export_path": "/exports/pong/",
    "n_envs": 8,  # Number of parallel environments
    
    # PPO hyperparameters optimized for Atari Pong
    "n_steps": 128,  # Steps per env per update
    "batch_size": 256,  # Minibatch size
    "n_epochs": 4,  # Number of epochs per update (this is what we'll call "epochs")
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.1,  # PPO clip range
    "ent_coef": 0.01,  # Entropy coefficient
    "vf_coef": 0.5,  # Value function coefficient
    "max_grad_norm": 0.5,  # Gradient clipping
    "learning_rate": 2.5e-4,  # Learning rate
    
    # Evaluation and checkpointing
    "eval_freq": 10000,  # Evaluate every 10k steps
    "n_eval_episodes": 10,  # Number of episodes for evaluation
    "save_freq_epochs": 10,  # Save checkpoint every N epochs (PPO updates)
}


# ------------------ ENVIRONMENT SETUP ------------------
def make_vec_env(env_name, n_envs, seed=0):
    """Create vectorized environment with proper seeding and monitoring."""
    def _make_env(rank):
        def _init():
            env = make_env(env_name)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    vec_env = SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    return vec_env


# ------------------ TRAINING ------------------
def train_model():
    """
    Train PPO agent on Pong following Part 2 requirements:
    - Uses exact preprocessing from PettingZoo notebook
    - Saves model every 10 epochs
    - Tracks and saves best model
    - Logs everything to Wandb
    """
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
        name=f"PPO_Pong_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        sync_tensorboard=True,
        save_code=True,
        tags=["pong", "ppo", "tournament-ready"]
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING PPO ON PONG - PART 2 (Tournament Ready)")
    print(f"{'='*70}")
    print(f"Environment: {config['env_name']}")
    print(f"Preprocessing: PettingZoo tournament specification")
    print(f"Observation space: Box(0.0, 1.0, (4, 84, 84), float32)")
    print(f"Action space: Discrete(6)")
    print(f"{'='*70}\n")

    # Create training environment
    print("Creating training environments...")
    env = make_vec_env(config["env_name"], config["n_envs"], seed=42)
    
    # Create evaluation environment
    print("Creating evaluation environment...")
    eval_env = make_vec_env(config["env_name"], n_envs=1, seed=123)
    
    # TensorBoard logging
    log_path = "./logs/pong_part2"
    os.makedirs(log_path, exist_ok=True)
    print(f"TensorBoard logs: {log_path}\n")
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        config["policy_type"],
        env,
        verbose=1,
        device=DEVICE,
        tensorboard_log=log_path,
        # !!! KEY FIX: Disable automatic normalization because our data is already float32 [0,1]
        policy_kwargs={"normalize_images": False},
        
        # Training hyperparameters
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
        learning_rate=config["learning_rate"],
    )
    
    print(f"\n{'='*70}")
    print("MODEL ARCHITECTURE:")
    print(f"{'='*70}")
    print(model.policy)
    print(f"{'='*70}\n")
    
    # Calculate frequencies
    steps_per_update = config["n_steps"] * config["n_envs"]  # This is one "epoch" in PPO terms
    checkpoint_freq_timesteps = steps_per_update * config["save_freq_epochs"]
    total_updates = config["total_timesteps"] // steps_per_update
    
    # Setup callbacks
    print("Setting up callbacks...")
    
    # 1. Evaluation callback - saves best model
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
    
    # 2. Checkpoint callback - saves model every N epochs
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq_timesteps,
        save_path=checkpoint_path,
        name_prefix="ppo_pong_epoch",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    # 3. Wandb callback for training metrics
    wandb_callback = WandbCallback(verbose=0)
    
    # 4. Best model to Wandb callback
    wandb_best_callback = WandbBestModelCallback(
        check_freq=config["eval_freq"],
        save_path=best_model_path,
        verbose=1
    )
    
    callback_list = CallbackList([
        wandb_callback,
        eval_callback,
        checkpoint_callback,
        wandb_best_callback
    ])
    
    # Training summary
    print(f"\n{'='*70}")
    print("TRAINING CONFIGURATION:")
    print(f"{'='*70}")
    print(f"Total timesteps:        {config['total_timesteps']:,}")
    print(f"Steps per update:       {steps_per_update:,}")
    print(f"Total updates (epochs): {total_updates:,}")
    print(f"Checkpoint frequency:   Every {config['save_freq_epochs']} epochs ({checkpoint_freq_timesteps:,} timesteps)")
    print(f"Evaluation frequency:   Every {config['eval_freq']:,} timesteps")
    print(f"Parallel environments:  {config['n_envs']}")
    print(f"Device:                 {DEVICE}")
    print(f"{'='*70}\n")
    
    print("Starting training...\n")
    print("Checkpoints will be saved to:")
    print(f"  - Every {config['save_freq_epochs']} epochs: {checkpoint_path}/")
    print(f"  - Best model: {best_model_path}/best_model.zip")
    print(f"  - Wandb: Automatic upload when new best found\n")
    
    t_start = datetime.now()
    
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback_list,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\n\n" + "="*70)
        print("⚠️  TRAINING INTERRUPTED BY USER")
        print("="*70)
        print("Saving current model state...")
    
    t_end = datetime.now()
    
    print(f"\n{'='*70}")
    print(f"✓ TRAINING COMPLETED")
    print(f"{'='*70}")
    print(f"Training time: {t_end - t_start}")
    print(f"{'='*70}\n")
    
    # Save final model
    final_path = os.path.join(config["export_path"], "ppo_pong_final")
    model.save(final_path)
    print(f"Final model saved: {final_path}.zip")
    
    # Save final model to Wandb
    wandb.save(f"{final_path}.zip", base_path=config["export_path"])
    print(f"Final model uploaded to Wandb\n")
    
    # Close environments
    env.close()
    eval_env.close()
    
    # Log final summary
    print(f"{'='*70}")
    print("TRAINING ARTIFACTS:")
    print(f"{'='*70}")
    print(f"Best model:      {best_model_path}/best_model.zip")
    print(f"Final model:     {final_path}.zip")
    print(f"Checkpoints:     {checkpoint_path}/")
    print(f"Wandb run:       {run.url}")
    print(f"{'='*70}\n")
    
    run.finish()
    
    return best_model_path + "/best_model"


# ------------------ EVALUATION ------------------
def evaluate_model(model_path, n_episodes=100, deterministic=True):
    """
    Evaluate a trained model in a single-player Pong environment.
    This matches Part 2 requirement: "Report the results of your agent 
    in a single-player environment (Pong from Gymnasium or Gym)."
    """
    print(f"\n{'='*70}")
    print(f"EVALUATING MODEL - SINGLE PLAYER PONG")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"Deterministic: {deterministic}")
    print(f"{'='*70}\n")
    
    # Load model
    if not model_path.endswith('.zip'):
        model_path = f"{model_path}.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return None, None
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device=DEVICE)
    print("✓ Model loaded successfully\n")
    
    # Create evaluation environment
    eval_env = make_env(config["env_name"])
    eval_env = Monitor(eval_env)
    
    print(f"Running {n_episodes} episodes...")
    print("-"*70)
    
    # Detailed evaluation with episode-by-episode results
    episode_rewards = []
    episode_lengths = []
    wins = 0
    losses = 0
    
    for episode in range(n_episodes):
        obs, _ = eval_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, terminated, truncated, info = eval_env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        
        # In Pong, positive score means win, negative means loss
        if episode_reward > 0:
            wins += 1
        elif episode_reward < 0:
            losses += 1
        
        if (episode + 1) % 10 == 0:
            print(f"Episodes {episode + 1}/{n_episodes} completed...")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    mean_length = np.mean(episode_lengths)
    win_rate = wins / n_episodes * 100
    
    print("-"*70)
    print(f"\n{'='*70}")
    print(f"EVALUATION RESULTS ({n_episodes} episodes)")
    print(f"{'='*70}")
    print(f"Mean Reward:        {mean_reward:.2f} ± {std_reward:.2f}")
    print(f"Mean Episode Length: {mean_length:.1f} steps")
    print(f"Win Rate:           {win_rate:.1f}% ({wins}/{n_episodes})")
    print(f"Loss Rate:          {(losses/n_episodes)*100:.1f}% ({losses}/{n_episodes})")
    print(f"Draws:              {n_episodes - wins - losses}")
    print(f"{'='*70}\n")
    
    # Log to wandb if not already in a run
    if wandb.run is None:
        wandb.init(
            project="Pong_Part2_Tournament",
            name=f"evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            job_type="evaluation"
        )
        wandb.log({
            "eval/mean_reward": mean_reward,
            "eval/std_reward": std_reward,
            "eval/mean_length": mean_length,
            "eval/win_rate": win_rate,
            "eval/n_episodes": n_episodes
        })
        wandb.finish()
    
    eval_env.close()
    
    return mean_reward, std_reward


# ------------------ VISUALIZATION ------------------
def visualize_agent(model_path, n_episodes=5):
    """Watch the trained agent play Pong."""
    print(f"\n{'='*70}")
    print(f"VISUALIZING AGENT")
    print(f"{'='*70}")
    print(f"Model: {model_path}")
    print(f"Episodes: {n_episodes}")
    print(f"{'='*70}\n")
    
    if not model_path.endswith('.zip'):
        model_path = f"{model_path}.zip"
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device=DEVICE)
    print("✓ Model loaded successfully\n")
    
    # Create environment with rendering
    env = make_env(config["env_name"], render="human")
    action_names = env.unwrapped.get_action_meanings()
    
    print(f"Available actions: {action_names}\n")

    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"{'='*70}")
        print(f"Episode {episode + 1}/{n_episodes}")
        print(f"{'='*70}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            action_str = action_names[action]
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
            
            if reward != 0:
                print(f"  Step {steps}: {action_str} -> Reward={reward:+.0f}, Score={episode_reward:+.0f}")
        
        result = "WIN" if episode_reward > 0 else "LOSS" if episode_reward < 0 else "DRAW"
        print(f"\nEpisode {episode + 1} Result: {result}")
        print(f"  Total Steps: {steps}")
        print(f"  Final Score: {episode_reward:+.0f}")
        print(f"{'='*70}\n")
    
    env.close()


# ------------------ MAIN ------------------
if __name__ == "__main__":
    """
    Part 2 Requirements Implementation:
    
    1. Select one RL model: ✓ PPO
    2. Tune parameters: ✓ Optimized config for Pong
    3. Train agent: ✓ With evaluation and checkpointing
    4. Report results in single-player: ✓ evaluate_model()
    5. Export trained agent: ✓ Saves to best_model/ and checkpoints/
    6. Test in 100 episodes: ✓ Use --eval with default 100 episodes
    7. Export video: Use --visualize (manual recording needed)
    
    Usage:
        Train:      python main_pong.py --train
        Evaluate:   python main_pong.py --eval [--model-path PATH] [--n-episodes 100]
        Visualize:  python main_pong.py --visualize [--model-path PATH] [--n-episodes 5]
        All:        python main_pong.py --train --eval
    """
    import argparse
    import numpy as np
    
    parser = argparse.ArgumentParser(
        description="Train PPO on Pong-v5 (Part 2 - Tournament Ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train from scratch:
    python main_pong.py --train
  
  Train and evaluate:
    python main_pong.py --train --eval
  
  Evaluate best model (100 episodes):
    python main_pong.py --eval
  
  Evaluate specific checkpoint:
    python main_pong.py --eval --model-path ./exports/pong/checkpoints/ppo_pong_epoch_100_steps
  
  Watch agent play:
    python main_pong.py --visualize
        """
    )
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--visualize", action="store_true", help="Visualize the agent playing")
    parser.add_argument("--model-path", type=str, default=None, 
                       help="Path to model for eval/viz (default: best_model)")
    parser.add_argument("--n-episodes", type=int, default=None, 
                       help="Number of episodes (default: 100 for eval, 5 for viz)")
    parser.add_argument("--deterministic", action="store_true", default=True,
                       help="Use deterministic actions during evaluation (default: True)")
    
    args = parser.parse_args()
    
    if args.train:
        print("\n" + "="*70)
        print("STARTING TRAINING - PART 2 (Tournament Ready)")
        print("="*70 + "\n")
        
        model_path = train_model()
        
        # Optionally evaluate after training
        if args.eval:
            print("\n" + "="*70)
            print("EVALUATING TRAINED MODEL")
            print("="*70 + "\n")
            n_eval = args.n_episodes if args.n_episodes else 100
            evaluate_model(model_path, n_episodes=n_eval, deterministic=args.deterministic)
        
    elif args.eval:
        if args.model_path is None:
            args.model_path = os.path.join(config['export_path'], "best_model", "best_model")
            print(f"No model path specified, using best model: {args.model_path}\n")
        
        n_eval = args.n_episodes if args.n_episodes else 100
        evaluate_model(args.model_path, n_episodes=n_eval, deterministic=args.deterministic)
        
    elif args.visualize:
        if args.model_path is None:
            args.model_path = os.path.join(config['export_path'], "best_model", "best_model")
            print(f"No model path specified, using best model: {args.model_path}\n")
        
        n_viz = args.n_episodes if args.n_episodes else 5
        visualize_agent(args.model_path, n_episodes=n_viz)
        
    else:
        print("\n" + "="*70)
        print("NO ARGUMENTS PROVIDED")
        print("="*70)
        print("\nStarting training by default...")
        print("Use --help to see all available options\n")
        model_path = train_model()