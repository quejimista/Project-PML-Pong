from functions.preprocessing import *
from functions.models import *
from functions.Agent import *
from functions.Replay_buffer import *
from functions.utils import *
import wandb
import datetime
import torch 

NAME_ENV = "PongNoFrameskip-v4"

lr = 0.0001         # Standard DQN learning rate (from paper)
MEMORY_SIZE = 100000  # Buffer capacity
MAX_EPISODES = 5000   # Maximum number of episodes
EPSILON = 1.0         # Start with full exploration
EPSILON_DECAY = 0.995 # Slower decay for better exploration
MIN_EPSILON = 0.01    # Minimum exploration rate
GAMMA = 0.99          # Discount factor
BATCH_SIZE = 32       # Batch size
BURN_IN = 10000       # Initial random experiences
DNN_UPD = 8           # Update every 4 steps (more stable)
DNN_SYNC = 1000       # Target network sync frequency

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
    net = DQN(env, learning_rate=lr, device=device)

    # Creating the buffer 
    buffer = ReplayBuffer(capacity=MEMORY_SIZE, burn_in=BURN_IN)

    # Creating the agent
    agent = Agent(env, net=net, buffer=buffer, epsilon=EPSILON, 
                  eps_decay=EPSILON_DECAY, batch_size=BATCH_SIZE)


    wandb.login()

    run_name = f"training_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    wandb.init(
        project="Project_Paradigms",
        name=run_name,  
        config={
            "learning_rate": lr,
            "batch_size": BATCH_SIZE,
            "gamma": GAMMA,
            "epsilon_start": EPSILON,
            "epsilon_decay": EPSILON_DECAY,
            "min_epsilon": MIN_EPSILON,
            "dnn_update_freq": DNN_UPD,
            "dnn_sync_freq": DNN_SYNC,
            "device": str(device)
        }
    )

    print(">>> Training starts at", datetime.datetime.now())
    print(f">>> Hyperparameters:")
    print(f"    LR: {lr}, Batch: {BATCH_SIZE}, Gamma: {GAMMA}")
    print(f"    Epsilon: {EPSILON} -> {MIN_EPSILON} (decay: {EPSILON_DECAY})")
    print(f"    Update freq: {DNN_UPD}, Sync freq: {DNN_SYNC}")

    agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES,
                batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD,
                dnn_sync_frequency=DNN_SYNC)
    
    plot_training_results(agent) 
    wandb.finish()

    # Saving the trained model
    model_filename = f"{NAME_ENV}_epsilon{EPSILON}_lr{lr}.dat"
    torch.save(net.state_dict(), model_filename)
    print(f">>> Model saved as: {model_filename}")
    print(">>> Training ends at", datetime.datetime.now())

    print(">>> Training ends at ",datetime.datetime.now())