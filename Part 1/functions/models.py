import torch
import torch.nn as nn        
import torch.optim as optim 
# from torchsummary import summary
import numpy as np
import os
from functions.Replay_buffer import ReplayBuffer
from functions.utils import epsilon_soft_action
import torch.nn.functional as F


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class DQN(torch.nn.Module):
    
    def __init__(self, env, learning_rate=1e-3, device='cpu'):
        super(DQN, self).__init__()
        self.device = device
        self.n_inputs = env.observation_space.shape
        self.n_outputs = env.action_space.n
        self.actions = np.arange(env.action_space.n)
        self.learning_rate = learning_rate
        
        ### Construction of the neural network
        self.net = nn.Sequential(
            nn.Conv2d(self.n_inputs[0], 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*7*7, 512),
            nn.ReLU(),
            nn.Linear(512, self.n_outputs)
        )
        
        self.optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        ### Work with CUDA is allowed
        if self.device == 'cuda':
            self.net.cuda()
    
    def forward(self, x):
        return self.net(x)
            
    
    def get_action(self, state, epsilon=0.05):
        """
        e-greedy method
        """
        if np.random.random() < epsilon:
            # random action
            action = np.random.choice(self.actions)  
        else:
            # Q-value based action
            qvals = self.get_qvals(state)  
            action= torch.max(qvals, dim=-1)[1].item()
        
        return action
    
    
    def get_qvals(self, state):
        if type(state) is tuple:
            state = np.array([np.ravel(s) for s in state])
        
        state_t = torch.FloatTensor(state).to(device=self.device)
        
        return self.net(state_t)
    





#original DQN network from the paper
def make_DQN(input_shape, output_shape):
    net = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, output_shape)
    )
    return net

class DoubleDQNAgent:
    def __init__(self, input_shape, n_actions, device, epsilon_scheduler=None):
        self.device = device  # cpu, gpu, etc
        self.n_actions = n_actions  # number of actions in the environment

        # networks
        self.q_net = make_DQN(input_shape, n_actions).to(device)  # primary network Q
        self.target_net = make_DQN(input_shape, n_actions).to(device)  # target network Q^
        self.target_net.load_state_dict(self.q_net.state_dict())  # they both start with the same weights
        self.target_net.eval()  # we train on the primary network not the target

        self.optimizer = optim.Adam(self.q_net.parameters(), lr=1e-4)  # adjust primary network weights

        self.gamma = 0.99
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(100000)

        #epsilon
        self.epsilon_scheduler = epsilon_scheduler
        if self.epsilon_scheduler is None:
            #fallback: classical exponential decay
            self.epsilon = 1.0
            self.epsilon_min = 0.1
            self.epsilon_decay = 0.999995

        self.update_target_freq = 1000  # C, how often to update the target network

        self.step_count = 0

    #select action, but using a epsilon soft-policy
    def select_action(self, state):
        if hasattr(self, 'epsilon_scheduler'):
            epsilon = self.epsilon_scheduler.get() #get current epsilon
        else:
            epsilon = self.epsilon 

        return epsilon_soft_action(self.q_net, state, self.n_actions, epsilon, self.device)

    #agent training
    def train_step(self):
        if len(self.replay_buffer) < self.batch_size: #dont start until enough samples
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size) #sample from buffer
        states, actions, rewards, next_states, dones = (
            #move all to device
            states.to(self.device),
            actions.to(self.device),
            rewards.to(self.device),
            next_states.to(self.device),
            dones.to(self.device),
        )

        #current Q(s,a)
        q_values = self.q_net(states).gather(1, actions)

        #target for double dqn
        with torch.no_grad():
            #chosen action for primary network
            next_actions = self.q_net(next_states).argmax(1, keepdim=True)
            #evaluated values by target network
            next_q_values = self.target_net(next_states).gather(1, next_actions)
            target = rewards + self.gamma * (1 - dones) * next_q_values

        #compute loss
        loss = F.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        #update epsilon
        if hasattr(self, 'epsilon_scheduler'):
            self.epsilon_scheduler.step() #get value from the scheduler passed
        else:
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay) #in case no epsilon scheduler, do exponential decay

        #every C steps, copy Q to Q^ (copy primary to target)
        self.step_count += 1
        if self.step_count % self.update_target_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()
    
    #saving function
    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_state_dict': self.q_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
            #we save the state dict of the primary network and the target network and the optimizer
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data['q_state_dict'])
        self.target_net.load_state_dict(data['target_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state'])