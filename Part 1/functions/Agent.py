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

        self.training_loss_history = []
        self.epsilon_history = []


    @torch.no_grad()
    def play_step(self, mode : str = 'train', device: torch.device = 'cpu', epsilon: float = 0.0):
        done_reward = None

        if np.random.random() < epsilon or mode =='explore':
            action = self.env.action_space.sample()
            print("Exploration mode or epsilon --> random action= ", action)
        else:
            state_v = torch.as_tensor(self.state).to(device)
            state_v.unsqueeze_(0)
            print(f'Size state_v = {state_v.shape()}')
            q_vals_v = self.net(state_v) # getting all the q values of that state
            print(f"Q values of state are {q_vals_v}")
            _, act_v = torch.max(q_vals_v, dim=1) # selecting the maximum value
            action = int(act_v.item())
            print(f'Action taken {action}')
            self.step_count += 1

        print(f'Doing step...\n')
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
            self.play_step(epsilon=self.epsilon, mode='explore')
 
        episode = 0
        training = True
        print("Training...")
        while training:
            self.state = self.env.reset()[0]
            self.total_reward = 0
            gamedone = False
            while gamedone == False:
                # The agent takes an action
                gamedone = self.play_step(epsilon=self.epsilon, mode='train')
               
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
                    # 1. Save Epsilon
                    self.epsilon_history.append(self.epsilon)
                    
                    # 2. Calculate and Save Average Loss for this episode
                    if len(self.update_loss) > 0:
                        avg_loss = np.mean(self.update_loss)
                        self.training_loss_history.append(avg_loss)
                    else:
                        self.training_loss_history.append(0) # No training steps this episode
                    
                    # LOGGING
                    wandb.log({
                        "episode": episode,
                        "reward": self.total_reward,
                        "mean_reward": mean_rewards,
                        "avg_loss": avg_loss,  # Variable calculated in Step 1
                        "epsilon": self.epsilon
                    })
                    print(f"Episode: {episode} | "
                          f"Steps: {self.step_count} | "
                          f"Reward: {self.total_reward:.2f} | "
                          f"Avg Reward: {mean_rewards:.2f} | "
                          f"Loss: {avg_loss:.5f} | "
                          f"Epsilon: {self.epsilon:.3f}")
                    print("\nLogging data to wandb...\n")

                    # 3. NOW reset the temp loss list
                    self.update_loss = []
                    
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
        states, actions, rewards, next_states, dones = [i for i in batch] 

        rewards_vals = torch.FloatTensor(rewards).to(device=self.net.device) 
        actions_vals = torch.LongTensor(np.array(actions)).reshape(-1,1).to(device=self.net.device)

        # Convert dones (floats) to boolean mask
        # dones_t = torch.tensor([bool(d) for d in dones], dtype=torch.bool, device=self.net.device)
        dones_t = dones.flatten().bool().to(device=self.net.device)

        # Obtain the Q values of the main network
        qvals = torch.gather(self.net.get_qvals(states), 1, actions_vals)
        
        # Obtain the target Q values.
        # The detach() parameter prevents these values from updating the target network
        qvals_next = torch.max(self.target_network.get_qvals(next_states), dim=-1)[0].detach().unsqueeze(1) # shape [32,1]
        # 0 in terminal states
        # qvals_next[dones_t] = 0 
        
        # Apply terminal mask to qvals_next
        qvals_next = qvals_next * (1 - dones) # dones is now the float tensor [32, 1]

        # Calculate the Bellman equation
        # Now, [32, 1] + [32, 1] works without broadcasting
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