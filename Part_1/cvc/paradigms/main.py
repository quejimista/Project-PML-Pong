from preprocessing import *
from models import *
from Agent import *
from Replay_buffer import *
from utils import *
import wandb
import datetime
import torch 
import sys

sys.stdout.reconfigure(line_buffering=True)


NAME_ENV = "PongNoFrameskip-v4"

LR = 0.0001         # Standard DQN learning rate (from paper)
MEMORY_SIZE = 100000  # Buffer capacity
MAX_EPISODES = 5000   # Maximum number of episodes
EPSILON = 1.0         # Start with full exploration
EPSILON_DECAY = 0.99 # Slower decay for better exploration
MIN_EPSILON = 0.01    # Minimum exploration rate
GAMMA = 0.99          # Discount factor
BATCH_SIZE = 64       # Batch size
BURN_IN = 10000       # Initial random experiences
DNN_UPD = 4           # Update every 4 steps (more stable)
DNN_SYNC = 1000       # Target network sync frequency
MODEL_LOADED = None

MODEL_TYPE = "DQN" # "DQN" "DoubleDQN"

# Prioritized Replay Settings
USE_PRIORITIZED_REPLAY = False  # Set to True to use prioritized replay
ALPHA = 0.6                     # Prioritization exponent (0 = uniform, 1 = full prioritization)
BETA_START = 0.4                # Initial importance sampling weight (0 = no correction applied, 1 = bias is corrected)
BETA_FRAMES = 500000            # Number of frames to anneal beta to 1.0


if __name__ == "__main__":
    # 1. DEVICE DETECTION
    if torch.cuda.is_available():
        device = torch.device("cuda")
        print(">>> CUDA Device Detected: ", torch.cuda.get_device_name(0))
    else:
        device = torch.device("cpu")
        print(">>> Device: CPU")

    # Creating the environment
    env = make_env(NAME_ENV)
    print_env_info("Wrapped", env)

    # Creating the network - PASS THE DETECTED DEVICE
    net = DQN(env, learning_rate=LR, device=device)

    # Creating the buffer 
    if USE_PRIORITIZED_REPLAY:
        buffer = PrioritizedReplayBuffer(capacity=MEMORY_SIZE, burn_in=BURN_IN, alpha=ALPHA)
        print(f">>> Using Prioritized Experience Replay (alpha={ALPHA})")
    else:
        buffer = ReplayBuffer(capacity=MEMORY_SIZE, burn_in=BURN_IN)
        print(">>> Using Standard Experience Replay")

    # Creating the agent
    # agent = Agent(env, net=net, buffer=buffer, epsilon=EPSILON, 
                #   eps_decay=EPSILON_DECAY, batch_size=BATCH_SIZE, model_type = MODEL_TYPE)
    agent = Agent(env, net=net, buffer=buffer, epsilon=EPSILON, 
                  eps_decay=EPSILON_DECAY, batch_size=BATCH_SIZE, 
                  model_type=MODEL_TYPE,
                  use_prioritized_replay=USE_PRIORITIZED_REPLAY,
                  beta_start=BETA_START,
                  beta_frames=BETA_FRAMES)


    wandb.login()

    # run_name = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_name = f"{MODEL_TYPE}_{'PER' if USE_PRIORITIZED_REPLAY else 'ER'}_lr{LR}_dnnupd{DNN_UPD}_epsdec{EPSILON_DECAY}_batch_{BATCH_SIZE}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

    
    wandb.init(
        project="Project_Paradigms",
        name=run_name,  
        config={
            "model_type": MODEL_TYPE,
            "use_prioritized_replay": USE_PRIORITIZED_REPLAY,
            "learning_rate": LR,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "epsilon_start": EPSILON,
            "epsilon_decay": EPSILON_DECAY,
            "min_epsilon": MIN_EPSILON,
            "dnn_update_freq": DNN_UPD,
            "dnn_sync_freq": DNN_SYNC,
            "buffer_capacity": MEMORY_SIZE,
            "burn_in": BURN_IN,
            "device": str(device)
        }
    )

    # Add prioritized replay specific config
    if USE_PRIORITIZED_REPLAY:
        wandb.config.update({
            "alpha": ALPHA,
            "beta_start": BETA_START,
            "beta_frames": BETA_FRAMES
        })

    print(">>> Training starts at", datetime.datetime.now())
    print(f">>> Hyperparameters:")
    print(f"    LR: {LR}, Batch: {BATCH_SIZE}, Gamma: {GAMMA}")
    print(f"    Epsilon: {EPSILON} -> {MIN_EPSILON} (decay: {EPSILON_DECAY})")
    print(f"    Update freq: {DNN_UPD}, Sync freq: {DNN_SYNC}")

    if MODEL_LOADED:
        if USE_PRIORITIZED_REPLAY:
            name_buffer = "PER"
        else:
            name_buffer = "ER"
        last_episode = agent.load_checkpoint(f'checkpoints/{MODEL_TYPE}_{name_buffer}_ep_1000.pt') #resume from checkpoint ===== CAMBIAR DEPENDIENDO DEL MODEL LOADED ==============================
    else:
        last_episode = 0
     
    agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES,
                batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD,
                dnn_sync_frequency=DNN_SYNC, resume_from_episode=last_episode)
        
    
    plot_training_results(agent) 
    wandb.finish()

    # Saving the trained model
    # model_filename = f"{NAME_ENV}_epsilon{EPSILON}_lr{lr}.dat"
    model_filename = f"{NAME_ENV}_{MODEL_TYPE}_{'PER' if USE_PRIORITIZED_REPLAY else 'ER'}_epsilon{EPSILON}_lr{LR}.dat"
    torch.save(net.state_dict(), model_filename)
    print(f">>> Model saved as: {model_filename}")
    print(">>> Training ends at", datetime.datetime.now())

    print(">>> Training ends at ",datetime.datetime.now())
