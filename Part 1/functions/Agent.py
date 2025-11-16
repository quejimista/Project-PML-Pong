import gymnasium as gym
from models import *
from functions.Replay_buffer import Experience, ReplayBuffer
import numpy as np
import torch
from copy import deepcopy, copy


class Agent:
    def __init__(self, env, net, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32):
        self.env = env
        self.net = net
        self.target_network = deepcopy(net) # red objetivo (copia de la principal)
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        # block of the last X episodes to calculate the average reward 
        self.nblock = 100 
        # average reward used to determine if the agent has learned to play
        self.reward_threshold = self.env.spec.reward_threshold 
        
        self.initialize()
    
    
    def initialize(self):
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.state = self.env.reset()[0]


    @torch.no_grad()
    def play_step(self, mode : str = 'train', device: torch.device = 'cpu', epsilon: float = 0.0):
        done_reward = None

        if np.random.random() < epsilon or mode =='explore':
            action = self.env.action_space.sample()
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            q_vals_v = self.net(state_v) # getting all the q values of that state
            _, act_v = torch.max(q_vals_v, dim=1) # selecting the maximum value
            action = int(act_v.item())
            self.step_count += 1

        # do step in the environment
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += float(reward)

        exp = Experience(state=self.state, action=action, reward=float(reward),
            done=is_done, new_state=new_state )
        
        self.buffer.append(exp)
        self.state = new_state
        if is_done:
            done_reward = self.total_reward
            self.state = self.env.reset()[0]
        return done_reward
    
    def train(self, gamma=0.99, max_episodes=50000, 
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000):
        self.gamma = gamma

        # Fill the buffer with N random experiences
        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.play_step(self.epsilon, mode='explore')
 
        episode = 0
        training = True
        print("Training...")
        while training:
            self.state = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                # The agent takes an action
                gamedone = self.play_step(self.epsilon, mode='train')
               
                # Upgrade main network
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                # Synchronize the main network and the target network
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.net.state_dict())
                    self.sync_eps.append(episode)
                    
                if gamedone:                   
                    episode += 1
                    # Save the rewards
                    self.training_rewards.append(self.total_reward) 
                    self.update_loss = []
                    # Calculate the average reward for the last X episodes
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)

                    print("\rEpisode {:d} Mean Rewards {:.2f} Epsilon {}\t\t".format(episode, mean_rewards, self.epsilon), end="")
                    self.display_training_progress(episode)

                    # Check if there are still episodes left
                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    # The game ends if the average reward has reached the threshold
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(episode))
                        break
                    
                    # Update epsilon according to the fixed decay rate
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
    
    ## Loss calculation           
    def calculate_loss(self, batch):
        # Separate the variables of the experience and convert them to tensors
        states, actions, rewards, dones, next_states = [i for i in batch] 
        rewards_vals = torch.FloatTensor(rewards).to(device=self.net.device) 
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(device=self.net.device)
        dones_t = torch.BoolTensor(dones).to(device=self.net.device)
        
        # Obtain the Q values of the main network
        qvals = torch.gather(self.net.get_qvals(states), 1, actions_vals)
        
        # Obtain the target Q values.
        # The detach() parameter prevents these values from updating the target network
        qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=-1)[0].detach()
        # 0 in terminal states
        qvals_next[dones_t] = 0 
        
        # Calculate the Bellman equation
        expected_qvals = self.gamma * qvals_next + rewards_vals
        
        # Calculate the loss
        loss = torch.nn.MSELoss()(qvals, expected_qvals.reshape(-1,1))
        return loss
    

    def update(self):
        # Remove any gradient
        self.net.optimizer.zero_grad()  
        # Select a subset from the buffer
        batch = self.buffer.sample(batch_size=self.batch_size) 
        # Calculate the loss
        loss = self.calculate_loss(batch) 
        # Difference to get the gradients
        loss.backward() 
        # Apply the gradients to the neural network
        self.net.optimizer.step() 
        # Save loss values
        if self.net.device == 'cuda':
            self.update_loss.append(loss.detach().cpu().numpy())
        else:
            self.update_loss.append(loss.detach().numpy())


    def display_training_progress(self, episode):
        if len(self.training_rewards) == 0:
            return

        current_reward = self.training_rewards[-1]
        avg_reward = self.mean_training_rewards[-1]
        threshold = self.reward_threshold

        print(
            f"\rEpisode: {episode} | "
            f"Reward: {current_reward:.2f} | "
            f"Avg({self.nblock}): {avg_reward:.2f} | "
            f"Threshold: {threshold:.2f} | "
            f"Epsilon: {self.epsilon:.3f}",
            end=""
        )