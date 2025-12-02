import gymnasium as gym
import wandb
import torch
from datetime import datetime
from stable_baselines3 import PPO, DQN, A2C
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import CallbackList, EvalCallback
from stable_baselines3.common.evaluation import evaluate_policy
from functions.preprocessing import make_env

# ------------------ DEVICE ------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {DEVICE}")

# ------------------ CONFIG ------------------
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 2_000_000,
    "env_name": "ALE/Skiing-v5",
    "export_path": "./exports/",
}

models_to_train = ["ppo"]  # Puedes agregar: "dqn", "a2c"

# ------------------ Wandb reward callback ------------------
from stable_baselines3.common.callbacks import BaseCallback

class WandbRewardCallback(BaseCallback):
    """Log reward and episode length to Wandb each episode."""
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            ep_info = info.get("episode")
            if ep_info:
                wandb.log({
                    "reward": ep_info['r'],
                    "episode_length": ep_info['l'],
                    "step": self.num_timesteps
                })
        return True

# ------------------ ENVIRONMENT ------------------
print(f"Gymnasium version: {gym.__version__}")

# Vectorized environments
env = DummyVecEnv([lambda: make_env(config["env_name"]) for _ in range(8)])
eval_env = make_env(config["env_name"])
eval_env = Monitor(eval_env)

reward_threshold = -5000  # puedes ajustar según Skiing-v5

# ------------------ TRAINING ------------------
def train_model(env, model_name, thresh):
    run = wandb.init(
        project="Paradigms_Part3",
        config=config,
        name=f"{model_name}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        sync_tensorboard=True,
        save_code=True
    )

    print(f"\n>>> Training model '{model_name}'...")

    # Crear el modelo
    if model_name == "dqn":
        model = DQN(config["policy_type"], env, verbose=1, device=DEVICE, tensorboard_log=f"runs/{run.id}")
    elif model_name == "a2c":
        model = A2C(config["policy_type"], env, verbose=1, device=DEVICE, tensorboard_log=f"runs/{run.id}")
    elif model_name == "ppo":
        model = PPO(config["policy_type"], env, verbose=1, device=DEVICE, tensorboard_log=f"runs/{run.id}")
    else:
        raise ValueError(f"Unknown model: {model_name}")

    # Eval callback
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=f"{config['export_path']}best_{model_name}",
        log_path=f"./logs/{model_name}",
        eval_freq=5000,
        deterministic=True,
        render=False,
        verbose=1
    )

    # Callback list
    callback_list = CallbackList([WandbRewardCallback(), eval_callback])

    # Entrenamiento
    model.learn(total_timesteps=config["total_timesteps"], callback=callback_list)

    # Guardar modelo final
    model.save(config["export_path"] + model_name)
    print(f"Model saved at '{config['export_path'] + model_name}'")

    run.finish()


# ------------------ EVALUATION ------------------
def eval_model(env, model_name):
    print(f"\n>>> Evaluating model '{model_name}'...")
    if model_name == "dqn":
        model = DQN.load(config["export_path"] + model_name, device=DEVICE)
    elif model_name == "a2c":
        model = A2C.load(config["export_path"] + model_name, device=DEVICE)
    elif model_name == "ppo":
        model = PPO.load(config["export_path"] + model_name, device=DEVICE)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=20)
    print(f"Reward (mean ± std): {mean_reward:.2f} ± {std_reward:.2f}")
    wandb.init(project="Paradigms_Part3")  # log evaluation reward
    wandb.log({"eval_mean_reward": mean_reward, "eval_std_reward": std_reward})
    wandb.finish()


# ------------------ MAIN ------------------
for model_name in models_to_train:
    train_model(env, model_name, reward_threshold)
    eval_model(eval_env, model_name)

