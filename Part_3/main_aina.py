import wandb
import torch
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback
from stable_baselines3.common.evaluation import evaluate_policy
from preprocessing_aina import make_env


# ------------------ DEVICE ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")


# ------------------ Wandb reward callback ------------------
class WandbRewardCallback(BaseCallback):
    """Log reward and episode length to Wandb each episode."""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_count = 0

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                self.episode_count += 1
                wandb.log({
                    "train/reward": ep_info['r'],
                    "train/episode_length": ep_info['l'],
                    "train/timesteps": self.episode_count
                }, step=self.num_timesteps)
        return True


# ------------------ CONFIG ------------------
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 10_000_000,  # Skiing needs significant training
    "env_name": "ALE/Skiing-v5",
    "export_path": "./exports/",
    "n_envs": 8,  # Number of parallel environments
    
    # Optimized PPO hyperparameters for Atari
    "n_steps": 128,  # Steps per env per update
    "batch_size": 256,  # Minibatch size
    "n_epochs": 4,  # Number of epochs per update
    "gamma": 0.99,  # Discount factor
    "gae_lambda": 0.95,  # GAE lambda
    "clip_range": 0.2,  # PPO clip range
    "ent_coef": 0.1,  # Entropy coefficient for exploration
    "vf_coef": 1.0,  # Value function coefficient
    "max_grad_norm": 0.5,  # Gradient clipping
    "learning_rate": 3e-4,  # Learning rate
    "normalize_advantage": True,  # Normalize advantages
}


# ------------------ ENVIRONMENT SETUP ------------------
def make_vec_env(env_name, n_envs, seed=0):
    """Create vectorized environment with proper seeding."""
    def _make_env(rank):
        def _init():
            env = make_env(env_name)
            env = Monitor(env)
            env.reset(seed=seed + rank)
            return env
        return _init
    
    # Use SubprocVecEnv for better performance (parallel processes)
    # Use DummyVecEnv if you have issues with multiprocessing
    return SubprocVecEnv([_make_env(i) for i in range(n_envs)])


# ------------------ TRAINING ------------------
def train_model():
    run = wandb.init(
        project="Paradigms_Part3",
        config=config,
        name=f"PPO_trying_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        sync_tensorboard=True,
        save_code=True
    )
    
    print(f"\n{'='*60}")
    print(f"Training PPO on {config['env_name']}")
    print(f"{'='*60}\n")

    # Create training environment
    env = make_vec_env(config["env_name"], config["n_envs"], seed=42)
    
    # Create evaluation environment
    eval_env = make_env(config["env_name"])
    eval_env = Monitor(eval_env)
    
    # Create PPO model with optimized hyperparameters
    model = PPO(
        config["policy_type"],
        env,
        verbose=1,
        device=DEVICE,
        tensorboard_log=f"runs/{run.id}",
        
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
    
    print(f"\nModel architecture:")
    print(model.policy)
    
    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{config['export_path']}best_ppo_skiing",
        log_path=f"./logs/ppo_skiing",
        eval_freq=10000,  # Evaluate every 10k steps
        n_eval_episodes=10,
        deterministic=False, 
        render=False,
        verbose=1
    )
    
    callback_list = CallbackList([
        WandbRewardCallback(verbose=0),
        eval_callback
    ])
    
    # Train
    print(f"\nStarting training for {config['total_timesteps']:,} timesteps...")
    print(f"This equals {config['total_timesteps'] // (config['n_steps'] * config['n_envs']):,} updates")
    print(f"Evaluation every {10000 // config['n_envs']} updates\n")
    
    t_start = datetime.now()
    model.learn(
        total_timesteps=config["total_timesteps"],
        callback=callback_list,
        progress_bar=True
    )
    t_end = datetime.now()
    
    print(f"\n{'='*60}")
    print(f"Training completed in: {t_end - t_start}")
    print(f"{'='*60}\n")
    
    # Save final model
    final_path = f"{config['export_path']}ppo_skiing_final"
    model.save(final_path)
    print(f"Final model saved to: {final_path}")
    
    # Close environments
    env.close()
    eval_env.close()
    
    run.finish()
    
    return final_path


# ------------------ EVALUATION ------------------
def evaluate_model(model_path, n_episodes=100):
    print(f"\n{'='*60}")
    print(f"Evaluating model: {model_path}")
    print(f"{'='*60}\n")
    
    # Load model
    model = PPO.load(model_path, device=DEVICE)
    
    # Create evaluation environment
    eval_env = make_env(config["env_name"])
    eval_env = Monitor(eval_env)
    
    # Evaluate
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_episodes,
        deterministic=True
    )
    
    print(f"\nEvaluation Results ({n_episodes} episodes):")
    print(f"Mean Reward: {mean_reward:.2f} ± {std_reward:.2f}")
    
    # Log to wandb
    wandb.init(project="Paradigms_Part3_Skiing", name="evaluation")
    wandb.log({
        "eval/mean_reward": mean_reward,
        "eval/std_reward": std_reward,
        "eval/n_episodes": n_episodes
    })
    wandb.finish()
    
    eval_env.close()
    
    return mean_reward, std_reward


# ------------------ VISUALIZATION ------------------
def visualize_agent(model_path, n_episodes=5):
    """Watch the trained agent play."""
    print(f"\n{'='*60}")
    print(f"Visualizing agent: {model_path}")
    print(f"{'='*60}\n")
    
    model = PPO.load(model_path, device=DEVICE)
    
    # Create environment with rendering
    env = make_env(config["env_name"], render="human")
    
    for episode in range(n_episodes):
        obs, info = env.reset()
        episode_reward = 0
        done = False
        steps = 0
        
        print(f"\nEpisode {episode + 1}/{n_episodes}")
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            done = terminated or truncated
        
        print(f"  Steps: {steps}")
        print(f"  Reward: {episode_reward:.2f}")
    
    env.close()


# ------------------ MAIN ------------------
if __name__ == "__main__":
     # training command: python main_aina.py --train
     # evaluating trained model: python main_aina.py --eval --model-path ./exports/best_ppo_skiing/best_model.zip
     #watch agent play: python main_aina.py --visualize --model-path ./exports/best_ppo_skiing/best_model.zip

    import argparse
    
    parser = argparse.ArgumentParser(description="Train PPO on Skiing-v5")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--eval", action="store_true", help="Evaluate the model")
    parser.add_argument("--visualize", action="store_true", help="Visualize the agent")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model for eval/viz")
    
    args = parser.parse_args()
    
    if args.train:
        model_path = train_model()
        
        # Optionally evaluate after training
        print("\nEvaluating trained model...")
        evaluate_model(model_path, n_episodes=50)
        
    elif args.eval:
        if args.model_path is None:
            args.model_path = f"{config['export_path']}best_ppo_skiing/best_model.zip"
        evaluate_model(args.model_path)
        
    elif args.visualize:
        if args.model_path is None:
            args.model_path = f"{config['export_path']}best_ppo_skiing/best_model.zip"
        visualize_agent(args.model_path)
        
    else:
        # Default: just train
        print("No arguments provided. Starting training...")
        print("Use --train, --eval, or --visualize flags for specific actions")
        model_path = train_model()