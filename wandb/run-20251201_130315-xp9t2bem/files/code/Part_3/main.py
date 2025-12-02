import wandb
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecVideoRecorder
from datetime import datetime 
from stable_baselines3 import DQN, A2C, PPO
from stable_baselines3.ppo.policies import MlpPolicy
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, CallbackList
from wandb.integration.sb3 import WandbCallback
from stable_baselines3.common.evaluation import evaluate_policy
import torch
from functions.preprocessing import make_env

def train_model(env, model_name, thresh):
    # Wandb setup
    run = wandb.init(
        project="Paradigms_Part3",
        config=config,
        name = f"{model_name}_date_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",      # set run name to model name
        sync_tensorboard=True,  # auto-upload sb3's tensorboard metrics
        save_code=True,         # optional
    )

    print("\n>>> Creating and training model '{}'...".format(model_name))
    
    # create
    if model_name == "dqn":
        model = DQN(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}", device=DEVICE)
    elif model_name == "a2c":
        model = A2C(config["policy_type"], env, verbose=0, tensorboard_log=f"runs/{run.id}", device=DEVICE)
    elif model_name == "ppo":
        model = PPO("CnnPolicy", env, verbose=1, device=DEVICE, n_steps=4096, 
                    batch_size=8192,gamma=0.9999)
    else:
        print("Error, unknown model ({})".format(model_name))
        return None
    
    ## Create eval environment and callback
    # Separate evaluation env
    eval_env = make_env()
    print("Environment reward threshold: {}".format(thresh))

    # Stop training when the model reaches the reward threshold
    #callback_on_best = StopTrainingOnRewardThreshold(reward_threshold=thresh, verbose=1) # specify the threshold of the environment
    eval_callback = EvalCallback(
        eval_env, 
        best_model_save_path=f"{config['export_path']}best_{model_name}", #save best model
        log_path=f"./logs/{model_name}",
        eval_freq=10000, #eval every 5000 steps
        deterministic=True, 
        render=False,
        verbose=1
    )
    # Create the callback list
    callback_list = CallbackList([WandbCallback(verbose=2), eval_callback])

    # train
    t0 = datetime.now() 
    model.learn(total_timesteps=config["total_timesteps"], callback=callback_list) # use the callback list in the learning function to be used during training
    t1 = datetime.now()
    print('>>> Training time (hh:mm:ss.ms): {}'.format(t1-t0))

    # save and export model
    model.save(config['export_path'] + model_name)
    print("Model exported at '{}'".format(config['export_path'] + model_name))

    # wandb
    run.finish()

def eval_model(env, model_name):
    print("Loading and evaluating model '{}'...".format(model_name))

    # load model
    if model_name == "dqn":
        model = DQN.load(config["export_path"] + model_name, device=DEVICE)
    elif model_name == "a2c":
        model = A2C.load(config["export_path"] + model_name, device=DEVICE)
    elif model_name == "ppo":
        model = PPO.load(config["export_path"] + model_name, device=DEVICE)
    else:
        print("Error, unknown model ({})".format(model_name), device=DEVICE)

    # evaluate the agent
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=100)
    print("Reward (mean +- std): {:.2f} +- {:.4f}".format(mean_reward, std_reward))






DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# ------------------------ MAIN -----------------------

print(f"Using device: {DEVICE}")

# CONFIGURATION
# list of models
models = ["ppo"]



# configuration file
config = {
    "policy_type": "CnnPolicy",
    "total_timesteps": 10000000,
    "env_name": "ALE/Skiing-v5",
    "export_path": "./exports/",
}


# ENVIRONMENT
print("Using Gymnasium version {}".format(gym.__version__))

env_name = config["env_name"]
env = make_env()
thresh = env.spec.reward_threshold if env.spec.reward_threshold is not None else -5000
# create environment
env = DummyVecEnv([lambda: make_env(env_name) for _ in range(8)])


# Training process
for model_name in models:
    train_model(env, model_name, thresh)


# Evaluation process
eval_env = make_env(env_name)

for model_name in models:
    eval_model(eval_env, model_name)