from functions.models import *
from functions.Replay_buffer import Experience, ReplayBuffer
import numpy as np
import torch
from copy import deepcopy
import wandb


class Agent:
    def __init__(self, env, net, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32, min_epsilon=0.01):
        self.env = env
        self.net = net
        self.target_network = deepcopy(net) 
        self.target_network.to(self.net.device) # Ensure target net is also on GPU
        self.buffer = buffer
        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.batch_size = batch_size
        self.min_epsilon = min_epsilon
        self.nblock = 100 
        self.reward_threshold = self.env.spec.reward_threshold if self.env.spec.reward_threshold is not None else 18.0 
        
        self.initialize()
    
    
    def initialize(self):
        self.update_loss = []
        self.training_rewards = []
        self.mean_training_rewards = []
        self.sync_eps = []
        self.total_reward = 0
        self.step_count = 0
        self.episode_step_count = 0
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
        self.episode_step_count += 1

        # do step in the environment
        new_state, reward, terminated, truncated, _ = self.env.step(action)
        is_done = terminated or truncated
        self.total_reward += float(reward)

        exp = Experience(state=self.state, action=action, reward=float(reward),
            done=is_done, new_state=new_state )
        
        self.buffer.append(exp)
        self.state = new_state

        if self.step_count % 500 == 0 and mode == 'train':
            mean_reward = (np.mean(self.training_rewards[-self.nblock:]) 
                              if len(self.training_rewards) > 0 else 0.0)
            
            print(f"Steps: {self.step_count} | "
                  f"Reward: {self.total_reward:.2f} | "
                  f"Mean reward: {mean_reward:.2f} | "
                  f"Eps: {self.epsilon:.3f}")
            
            wandb.log({
                "step": self.step_count,
                "current_reward": self.total_reward,
                "mean_reward": mean_reward,
                "epsilon": self.epsilon
            })

        # Handle episode end
        if is_done:
            done_reward = self.total_reward
            
        return done_reward
    
    def train(self, gamma=0.99, max_episodes=50000, 
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000):
        self.gamma = gamma

        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.play_step(epsilon=1.0, mode='explore')
        
        print(f"Buffer filled with {len(self.buffer)} experiences")
 
        episode = 0
        training = True
        print("Training...")
        while training:
            self.state = self.env.reset()[0]
            self.total_reward = 0
            self.episode_step_count = 0
            episode_done = False

            while not episode_done:
                # Play step with current epsilon
                reward_if_done = self.play_step(epsilon=self.epsilon, mode='train')
               
                # Update network
                if self.step_count % dnn_update_frequency == 0:
                    self.update()
                
                # Sync target network
                if self.step_count % dnn_sync_frequency == 0:
                    self.target_network.load_state_dict(self.net.state_dict())
                    self.sync_eps.append(episode)
                    print(f">>> Target network synced at step {self.step_count}")

                # Episode finished
                if reward_if_done is not None:   
                    episode_done = True  
                    episode += 1

                    final_reward = reward_if_done
                    self.training_rewards.append(final_reward)
                    self.epsilon_history.append(self.epsilon)
                    
                    # Calculate average loss
                    avg_loss = (np.mean(self.update_loss) 
                               if len(self.update_loss) > 0 else 0.0)
                    self.training_loss_history.append(avg_loss)
                    
                    # Calculate mean reward
                    mean_rewards = np.mean(self.training_rewards[-self.nblock:])
                    self.mean_training_rewards.append(mean_rewards)
                    
                    # Log to wandb
                    wandb.log({
                        "episode": episode,
                        "episode_reward": final_reward,
                        "episode_steps": self.episode_step_count,
                        "mean_reward_100": mean_rewards,
                        "avg_loss": avg_loss, 
                        "epsilon": self.epsilon
                    })

                    print(f"\n{'='*70}")
                    print(f"EPISODE {episode} COMPLETED")
                    print(f"{'='*70}")
                    print(f"Total Steps: {self.step_count} | "
                          f"Episode Steps: {self.episode_step_count}")
                    print(f"Episode Reward: {final_reward:.2f} | "
                          f"Mean(100): {mean_rewards:.2f}")
                    print(f"Loss: {avg_loss:.5f} | "
                          f"Epsilon: {self.epsilon:.3f}")
                    print(f"{'='*70}\n")

                    self.update_loss = []
                    
                    # Decay epsilon with minimum threshold
                    self.epsilon = max(self.epsilon * self.eps_decay, self.min_epsilon)
                    
                    # Check termination conditions
                    if episode >= max_episodes:
                        training = False
                        print('\n>>> Episode limit reached.')
                        break
                    
                    if mean_rewards >= self.reward_threshold:
                        training = False
                        print(f'\n>>> Environment solved in {episode} episodes!')
                        print(f'>>> Mean reward: {mean_rewards:.2f}')
                        break
    
    ## Loss calculation           
    def calculate_loss(self, batch):
        # Batch comes from buffer 
        states, actions, rewards, next_states, dones = batch
        
        # MOVE TO DEVICE 
        states = states.to(self.net.device)
        actions = actions.to(self.net.device)
        rewards = rewards.to(self.net.device)
        next_states = next_states.to(self.net.device)
        dones = dones.to(self.net.device)

        # Current Q-values
        qvals = torch.gather(self.net(states), 1, actions)
        
        # Target Q-values
        with torch.no_grad():
            qvals_next = torch.max(self.target_network(next_states), dim=-1)[0].unsqueeze(1)
        
        # Bellman equation: Target = R + gamma * Q_next * (1 - Done)
        expected_qvals = rewards + self.gamma * qvals_next * (1 - dones)
        
        loss = torch.nn.MSELoss()(qvals, expected_qvals)
        return loss
    

    def update(self):
        # Only update if buffer has enough samples
        if len(self.buffer) < self.batch_size:
            return
            
        self.net.optimizer.zero_grad()  
        batch = self.buffer.sample(batch_size=self.batch_size) 
        loss = self.calculate_loss(batch) 
        loss.backward() 
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        
        self.net.optimizer.step() 
        self.update_loss.append(loss.item())