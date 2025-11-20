from functions.preprocessing import *
from functions.models import *
from functions.Agent import *
from functions.Replay_buffer import *
from functions.utils import *
import wandb
import datetime
import torch 

NAME_ENV = "PongNoFrameskip-v4"

lr = 0.0001           # Learning rate (Low LR is usually better for Pong)
MEMORY_SIZE = 100000  # Maximum buffer capacity
MAX_EPISODES = 5000   # Maximum number of episodes
EPSILON = 1           # Initial value of epsilon
EPSILON_DECAY = 0.99  # epsilon decay
GAMMA = 0.99          # Gamma value of the Bellman equation
BATCH_SIZE = 32       # Number of elements to extract from the buffer
BURN_IN = 10000        # Number of initial episodes used to fill the buffer
DNN_UPD = 1           # Neural network update rate
DNN_SYNC = 1000       # Frequency of synchronization

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
    wandb.init(project="Project_Paradigms", config={
        "learning_rate": lr,
        "batch_size": BATCH_SIZE,
        "gamma": GAMMA,
        "device": str(device)
    })

    print(">>> Training starts at ",datetime.datetime.now())

    agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES,
                batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD,
                dnn_sync_frequency=DNN_SYNC)
    
    plot_training_results(agent) 
    wandb.finish()

    # Saving the trained model
    torch.save(net.state_dict(), NAME_ENV + ".dat")

    print(">>> Training ends at ",datetime.datetime.now())