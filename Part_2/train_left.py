import os
import numpy as np
import gymnasium as gym
import supersuit as ss
import wandb
import torch
from datetime import datetime
from pettingzoo.atari import pong_v3
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, BaseCallback, CheckpointCallback

# ------------------ DEVICE ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# Path to your ALREADY TRAINED Right-side agent
# UPDATE THIS PATH if it is different
RIGHT_AGENT_PATH = "./exports/pong/best_model/best_model.zip"


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


# ------------------ WANDB CALLBACKS ------------------
class WandbCallback(BaseCallback):
    """
    Custom callback for logging training metrics to Wandb.
    """
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
                    "train/episode_reward": ep_info['r'],
                    "train/episode_length": ep_info['l'],
                    "train/episode_count": self.episode_count
                }, step=self.num_timesteps)
        return True

    def _on_rollout_end(self) -> None:
        if hasattr(self.model, 'logger') and self.model.logger is not None:
            wandb.log({
                "train/learning_rate": self.model.learning_rate,
            }, step=self.num_timesteps)

class WandbBestModelCallback(BaseCallback):
    """
    Callback to save best model to Wandb when a new best is found.
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
                        data = np.load(eval_file)
                        if len(data['results']) > 0:
                            current_best = data['results'][-1].mean()
                            if current_best > self.best_mean_reward:
                                self.best_mean_reward = current_best
                                wandb.save(best_model_path, base_path=os.path.dirname(self.save_path))
                                if self.verbose > 0:
                                    print(f"\n{'='*70}")
                                    print(f"🏆 NEW BEST MODEL: {self.best_mean_reward:.2f}")
                                    print(f"{'='*70}\n")
                                wandb.log({"eval/best_mean_reward": self.best_mean_reward}, step=self.num_timesteps)
                except Exception as e:
                    if self.verbose > 0:
                        print(f"Error saving best model to Wandb: {e}")
        return True

# ------------------ CONFIG ------------------
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 5_000_000,
    "env_name": "PettingZoo/Left-vs-Right",
    "export_path": "./exports/pong_left/", # New export path for Left Agent
    "n_envs": 8,
    "opponent_path": RIGHT_AGENT_PATH,
    
    # Hyperparameters
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
    
    # Evaluation and checkpointing
    "eval_freq": 10000,
    "n_eval_episodes": 10,
    "save_freq_epochs": 10,
}

# ------------------ WRAPPER & ENV SETUP ------------------
class LeftPlayerWrapper(gym.Env):
    """
    Wraps PettingZoo to control the Left Player ('first_0').
    The Right Player ('second_0') is controlled by the pre-trained opponent.
    """
    def __init__(self, opponent_model_path):
        super().__init__()
        
        # 1. Initialize PettingZoo Pong
        self.pz_env = pong_v3.parallel_env(render_mode=None)
        
        # 2. Apply EXACT tournament preprocessing
        self.pz_env = ss.color_reduction_v0(self.pz_env, mode='B')
        self.pz_env = ss.resize_v1(self.pz_env, x_size=84, y_size=84)
        self.pz_env = ss.frame_stack_v1(self.pz_env, 4, stack_dim=0) # Channel First (4, 84, 84)
        self.pz_env = ss.dtype_v0(self.pz_env, dtype=np.float32)
        self.pz_env = ss.normalize_obs_v0(self.pz_env, env_min=0, env_max=1)
        self.pz_env = ss.reshape_v0(self.pz_env, (4, 84, 84))
        
        self.pz_env.reset()
        
        self.observation_space = self.pz_env.observation_space("first_0")
        self.action_space = self.pz_env.action_space("first_0")
        
        # Load Opponent (Right Agent) on CPU to save VRAM
        if os.path.exists(opponent_model_path):
            try:
                self.opponent = PPO.load(opponent_model_path, device="cpu")
            except Exception as e:
                print(f"Error loading opponent: {e}")
                self.opponent = None
        else:
            self.opponent = None

    def reset(self, seed=None, options=None):
        obs_dict, info = self.pz_env.reset(seed=seed, options=options)
        self.last_obs = obs_dict
        return obs_dict["first_0"], info["first_0"]

    def step(self, action):
        left_action = action
        
        # Opponent Logic (Right Side)
        if self.opponent:
            right_obs = self.last_obs["second_0"]
            right_action, _ = self.opponent.predict(right_obs, deterministic=True)
            if isinstance(right_action, np.ndarray):
                right_action = int(right_action)
        else:
            right_action = self.pz_env.action_space("second_0").sample()
            
        actions = {"first_0": left_action, "second_0": right_action}
        obs_dict, rewards, terms, truncs, infos = self.pz_env.step(actions)
        
        self.last_obs = obs_dict
        
        return (
            obs_dict["first_0"], 
            rewards["first_0"], 
            terms["first_0"], 
            truncs["first_0"], 
            infos["first_0"]
        )
    
    def close(self):
        self.pz_env.close()

def make_vec_env(opponent_path, n_envs, seed=0):
    """Create vectorized environment of LeftPlayerWrapper."""
    def _make_env(rank):
        def _init():
            env = LeftPlayerWrapper(opponent_path)
            # Use Monitor to ensure SB3 tracks episode stats
            env = Monitor(env) 
            env.reset(seed=seed + rank)
            return env
        return _init
    
    # Use SubprocVecEnv for true parallelism
    vec_env = SubprocVecEnv([_make_env(i) for i in range(n_envs)])
    vec_env = VecMonitor(vec_env)
    return vec_env

# ------------------ TRAINING ------------------
def train_model():
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
        name=f"PPO_LEFT_Agent_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        sync_tensorboard=True,
        save_code=True,
        tags=["pong", "left-agent", "vs-pretrained"]
    )
    
    print(f"\n{'='*70}")
    print(f"TRAINING LEFT AGENT vs RIGHT AGENT")
    print(f"{'='*70}")
    print(f"Opponent Path: {config['opponent_path']}")
    print(f"Export Path: {config['export_path']}")
    print(f"{'='*70}\n")

    # Create training environment
    print("Creating training environments...")
    env = make_vec_env(config["opponent_path"], config["n_envs"], seed=42)
    
    # Create evaluation environment (Against same opponent)
    print("Creating evaluation environment...")
    eval_env = make_vec_env(config["opponent_path"], n_envs=1, seed=123)
    
    # Create PPO model
    print("Initializing PPO model...")
    model = PPO(
        config["policy_type"],
        env,
        verbose=1,
        device=DEVICE,
        tensorboard_log="./logs/pong_left",
        policy_kwargs={"normalize_images": False}, # SuperSuit handled this
        
        learning_rate=config["learning_rate"],
        n_steps=config["n_steps"],
        batch_size=config["batch_size"],
        n_epochs=config["n_epochs"],
        gamma=config["gamma"],
        gae_lambda=config["gae_lambda"],
        clip_range=config["clip_range"],
        ent_coef=config["ent_coef"],
        vf_coef=config["vf_coef"],
        max_grad_norm=config["max_grad_norm"],
    )
    
    # Calculate frequencies
    steps_per_update = config["n_steps"] * config["n_envs"]
    checkpoint_freq_timesteps = steps_per_update * config["save_freq_epochs"]
    
    # Callbacks
    # Console Logger (Prints to terminal)
    console_callback = ConsoleLoggerCallback(check_freq=1000) # Print every 1000 steps

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_path,
        log_path=best_model_path,
        eval_freq=config["eval_freq"],
        n_eval_episodes=config["n_eval_episodes"],
        deterministic=True,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq_timesteps,
        save_path=checkpoint_path,
        name_prefix="ppo_left_epoch",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1
    )
    
    wandb_callback = WandbCallback(verbose=0)
    
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
    
    print("Starting training...\n")
    
    try:
        model.learn(
            total_timesteps=config["total_timesteps"],
            callback=callback_list,
            progress_bar=True
        )
    except KeyboardInterrupt:
        print("\nTRAINING INTERRUPTED BY USER. Saving current model...")
    
    # Save final model
    final_path = os.path.join(config["export_path"], "ppo_pong_left_final")
    model.save(final_path)
    print(f"Final model saved: {final_path}.zip")
    
    # Save to Wandb
    wandb.save(f"{final_path}.zip", base_path=config["export_path"])
    
    env.close()
    eval_env.close()
    run.finish()
    
    return best_model_path + "/best_model"

# ------------------ EVALUATION ------------------
def evaluate_model(model_path, n_episodes=100, deterministic=True):
    print(f"\n{'='*70}")
    print(f"EVALUATING LEFT AGENT")
    print(f"{'='*70}")
    
    if not model_path.endswith('.zip'):
        model_path = f"{model_path}.zip"
    
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path, device=DEVICE)
    
    # Use the LeftPlayerWrapper for evaluation
    eval_env = make_vec_env(config["opponent_path"], n_envs=1, seed=999)
    
    print(f"Running {n_episodes} episodes against Right Agent...")
    
    wins = 0
    losses = 0
    
    # We can iterate manually over the vec env
    obs = eval_env.reset()
    
    for i in range(n_episodes):
        done = False
        ep_reward = 0
        while not done:
            action, _ = model.predict(obs, deterministic=deterministic)
            obs, reward, done_array, info = eval_env.step(action)
            ep_reward += reward[0]
            done = done_array[0]
            
        if ep_reward > 0: wins += 1
        elif ep_reward < 0: losses += 1
        
        if (i+1) % 10 == 0:
            print(f"Episode {i+1} completed. Current Win Rate: {wins/(i+1)*100:.1f}%")
            
    print(f"Final Win Rate: {wins/n_episodes*100:.1f}%")
    eval_env.close()


# ------------------ MAIN ------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", action="store_true", help="Train the Left Agent")
    parser.add_argument("--eval", action="store_true", help="Evaluate the Left Agent")
    parser.add_argument("--model-path", type=str, default=None)
    parser.add_argument("--n-episodes", type=int, default=100)
    
    args = parser.parse_args()
    
    if args.train:
        model_path = train_model()
        if args.eval:
            evaluate_model(model_path, n_episodes=args.n_episodes)
            
    elif args.eval:
        if args.model_path is None:
            args.model_path = os.path.join(config['export_path'], "best_model", "best_model")
        evaluate_model(args.model_path, n_episodes=args.n_episodes)