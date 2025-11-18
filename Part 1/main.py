from functions.preprocessing import *
from functions.models import *
from functions.Agent import *
from functions.Replay_buffer import *
import wandb
import datetime


NAME_ENV = "PongNoFrameskip-v4"

lr = 0.001            # Learning rate
MEMORY_SIZE = 100000  # Maximum buffer capacity
MAX_EPISODES = 5000   # Maximum number of episodes (the agent must learn before reaching this value)
EPSILON = 1           # Initial value of epsilon
EPSILON_DECAY = 0.99   # epsilon decay
GAMMA = 0.99          # Gamma value of the Bellman equation
BATCH_SIZE = 32       # Number of elements to extract from the buffer
BURN_IN = 1000        # Number of initial episodes used to fill the buffer before training
DNN_UPD = 1           # Neural network update rate
DNN_SYNC = 2500       # Frequency of synchronization between the neural network and the target network


if __name__ == "__main__":
    # Creating the environment
    env = make_env(NAME_ENV)
    print_env_info("Wrapped", env)

    # Creating the network (in this case the basic DQN)
    net = DQN(env, learning_rate=lr, device='cpu')

    # Creating the buffer 
    buffer = ReplayBuffer(capacity=MEMORY_SIZE, burn_in=BURN_IN)

    # Creating the agent
    agent = Agent(env, net=net, buffer=buffer, epsilon=EPSILON, 
                  eps_decay=EPSILON_DECAY, batch_size=BATCH_SIZE)


    wandb.login()
    wandb.init(project="Project_Paradigms")

    print(">>> Training starts at ",datetime.datetime.now())

    agent.train(gamma=GAMMA, max_episodes=MAX_EPISODES,
                batch_size=BATCH_SIZE, dnn_update_frequency=DNN_UPD,
                dnn_sync_frequency=DNN_SYNC)

    # Saving the trained model
    torch.save(net.state_dict(), NAME_ENV + ".dat")

    print(">>> Training ends at ",datetime.datetime.now())

    # Finish the wandb run, necessary in notebooks
    wandb.finish()

    # obs, info = env.reset()
    # done = False

    # while not done:
    #     action = env.action_space.sample()  #random sample for now
    #     obs, reward, done, truncated, info = env.step(action)
