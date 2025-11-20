from functions.models import *
from functions.Replay_buffer import Experience, ReplayBuffer
import numpy as np
import torch
from copy import deepcopy, copy
import wandb


class Agent:
    def __init__(self, env, net, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32):
        self.env = env
        self.net = net
        self.target_network = deepcopy(net) 
        self.target_network.to(self.net.device) # Ensure target net is also on GPU
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.nblock = 100 
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

        self.training_loss_history = []
        self.epsilon_history = []


    @torch.no_grad()
    def play_step(self, mode : str = 'train', epsilon: float = 0.0):
        done_reward = None

        if mode == 'explore' or np.random.rand() < epsilon:
            action = self.env.action_space.sample()
        else:
            state_a = np.array(self.state, copy=False)
            state_v = torch.tensor(state_a).to(self.net.device).unsqueeze(0)
            q_vals_v = self.net(state_v) 
            _, act_v = torch.max(q_vals_v, dim=1) 
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

        if self.step_count % 500 == 0 and self.step_count > 0 and mode == 'train':
            print(f"Steps: {self.step_count} | Reward: {self.total_reward:.2f} | Eps: {self.epsilon:.3f}")
            # LOGGING
            wandb.log({
                "step": self.step_count,
                "reward": reward,
                "total_reward": self.total_reward,
                "mean_reward": np.mean(self.training_rewards[-self.nblock:]),
                "epsilon": self.epsilon
            })

        if is_done:
            done_reward = self.total_reward
            self.state = self.env.reset()[0]
        return done_reward
    
    def train(self, gamma=0.99, max_episodes=50000, 
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000):
        self.gamma = gamma

        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.play_step(epsilon=self.epsilon, mode='explore')
 
        episode = 0
        training = True
        print("Training...")
        while training:
            self.state = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                gamedone = self.play_step(epsilon=self.epsilon, mode='train')
               
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.net.state_dict())
                    self.sync_eps.append(episode)
                if gamedone:     
                    episode += 1
                    self.training_rewards.append(self.total_reward) 
                    self.epsilon_history.append(self.epsilon)
                    
                    if len(self.update_loss) > 0:
                        avg_loss = np.mean(self.update_loss)
                        self.training_loss_history.append(avg_loss)
                    else:
                        self.training_loss_history.append(0)
                    
                    # LOGGING
                    wandb.log({
                        "episode": episode,
                        "reward": self.total_reward,
                        "mean_reward": np.mean(self.training_rewards[-self.nblock:]),
                        "avg_loss": avg_loss, 
                        "epsilon": self.epsilon
                    })

                    print(f"Episode: {episode} | Steps: {self.step_count} | Reward: {self.total_reward:.2f} | Loss: {avg_loss:.5f} | Eps: {self.epsilon:.3f}")

                    self.update_loss = []
                    
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)

                    if episode >= max_episodes:
                        training = False
                        print('\nEpisode limit reached.')
                        break
                    
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print('\nEnvironment solved in {} episodes!'.format(episode))
                        break
                    
                    self.epsilon = max(self.epsilon * self.eps_decay, 0.01)
    
    ## Loss calculation           
    def calculate_loss(self, batch):
        # Batch comes from buffer (CPU tensors)
        states, actions, rewards, next_states, dones = batch
        
        # MOVE TO DEVICE (GPU)
        states = states.to(self.net.device)
        actions = actions.to(self.net.device)
        rewards = rewards.to(self.net.device)
        next_states = next_states.to(self.net.device)
        dones = dones.to(self.net.device)

        # Gather Q-values (Action is already correct shape [B, 1] from buffer)
        # actions is int64, gather requires int64
        qvals = torch.gather(self.net(states), 1, actions)
        
        # Target Q-values
        with torch.no_grad():
            qvals_next = torch.max(self.target_network(next_states), dim=-1)[0].unsqueeze(1)
        
        # Bellman equation: Target = R + gamma * Q_next * (1 - Done)
        expected_qvals = rewards + self.gamma * qvals_next * (1 - dones)
        
        loss = torch.nn.MSELoss()(qvals, expected_qvals)
        return loss
    

    def update(self):
        self.net.optimizer.zero_grad()  
        batch = self.buffer.sample(batch_size=self.batch_size) 
        loss = self.calculate_loss(batch) 
        loss.backward() 
        self.net.optimizer.step() 
        
        # Save loss (move to cpu for numpy conversion)
        self.update_loss.append(loss.item())