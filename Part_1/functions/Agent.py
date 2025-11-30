from functions.models import *
from functions.Replay_buffer import Experience, ReplayBuffer
import numpy as np
import torch
from copy import deepcopy
import wandb
import os
import glob
import warnings

warnings.filterwarnings('ignore', category=FutureWarning, module='torch')

class Agent:
    def __init__(self, env, net, buffer, epsilon=0.1, eps_decay=0.99, batch_size=32, min_epsilon=0.01, 
                 model_type='DQN', use_prioritized_replay=False, beta_start=0.4, beta_frames=100000):
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
        self.model_type = model_type

        # PER parameters
        self.use_prioritized_replay = use_prioritized_replay
        self.beta_start = beta_start
        self.beta_frames = beta_frames
        self.beta = beta_start
        
        self.initialize()
    
    
    def initialize(self, continue_from_checkpoint = False):
        if not continue_from_checkpoint:
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

        # clipped_reward = np.sign(reward)  # -1, 0, or +1
        exp = Experience(state=self.state, action=action, reward=float(reward),
                    done=is_done, new_state=new_state)
        
        self.buffer.append(exp)
        self.state = new_state

        # Handle episode end
        if is_done:
            done_reward = self.total_reward
            
        return done_reward
    
    
    def train(self, gamma=0.99, max_episodes=50000, 
              batch_size=32,
              dnn_update_frequency=4,
              dnn_sync_frequency=2000,
              save_frequency = 50,
              save_dir='checkpoints',
              resume_from_episode=0
              ):
        self.gamma = gamma

        print("Filling replay buffer...")
        while self.buffer.burn_in_capacity() < 1:
            self.play_step(epsilon=1.0, mode='explore')
        
        print(f"Buffer filled with {len(self.buffer)} experiences")
        self.check_buffer_diversity()
        episode = resume_from_episode
        
        training = True
        os.makedirs(save_dir, exist_ok=True) #create save directory
        
        buffer_type = "Prioritized" if self.use_prioritized_replay else "Standard"
        print(f"Training a {self.model_type} network with {buffer_type} Replay Buffer")

        if resume_from_episode > 0:
            print(f"Resuming training from episode {resume_from_episode}\n")

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
                    # print(f">>> Target network synced at step {self.step_count}")

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
                    log_dict = {
                        "episode": episode,
                        "episode_reward": final_reward,
                        "episode_steps": self.episode_step_count,
                        "mean_reward_episode": mean_rewards,
                        "avg_loss": avg_loss, 
                        "epsilon": self.epsilon
                    }
                    
                    if self.use_prioritized_replay:
                        log_dict["beta"] = self.beta
                    
                    wandb.log(log_dict)

                    print(f"EPISODE {episode} COMPLETED")
                    print(f"{'='*70}")
                    print(f"Total Steps: {self.step_count} | "
                          f"Episode Steps: {self.episode_step_count}")
                    print(f"Episode Reward: {final_reward:.2f} | "
                          f"Mean Reward: {mean_rewards:.2f}")
                    print(f"Loss: {avg_loss:.5f} | "
                          f"Epsilon: {self.epsilon:.3f}")
                    
                    if self.use_prioritized_replay:
                        print(f"Beta: {self.beta:.3f}")
                    print(f"\n\n")


                    if episode % save_frequency == 0:
                        self.save_N_checkpoints(episode, save_dir, keep_last_n=5)

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
                        self.save_N_checkpoints(episode, save_dir, keep_last_n=5)
                        print(f'>>> Mean reward: {mean_rewards:.2f}')

                        # Save the final, solved model's state dict to the WandB run directory
                        final_model_path = os.path.join(wandb.run.dir, "solved_model.pt")
                        torch.save(self.net.state_dict(), final_model_path)
                        
                        # Use wandb.save() to upload the file to the run
                        wandb.save(final_model_path)
                        
                        # (Optional: Also save existing local checkpoints just in case)
                        wandb.save(os.path.join(wandb.run.dir, "checkpoint*"))
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
        # loss = torch.nn.SmoothL1Loss()(qvals, expected_qvals) # Huber loss
        return loss
    

    def calculate_loss_doubleDQN(self, batch):
        states, actions, rewards, next_states, dones = batch
        
        states = states.to(self.net.device)
        actions = actions.to(self.net.device)
        rewards = rewards.to(self.net.device)
        next_states = next_states.to(self.net.device)
        dones = dones.to(self.net.device)
        
        # Current Q-values
        qvals = torch.gather(self.net(states), 1, actions)
    
        # Double DQN: separate action selection and evaluation
        with torch.no_grad():
            #online network selects actions
            next_actions = torch.argmax(self.net(next_states), dim=-1, keepdim=True)
            # Target network evaluates those actions
            qvals_next = torch.gather(self.target_network(next_states), 1, next_actions)
        
        expected_qvals = rewards + self.gamma * qvals_next * (1 - dones)
        loss = torch.nn.MSELoss()(qvals, expected_qvals)
        
        return loss
    
    ## Loss calculation with Prioritized Replay
    def calculate_loss_prioritized(self, batch):
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        states = states.to(self.net.device)
        actions = actions.to(self.net.device)
        rewards = rewards.to(self.net.device)
        next_states = next_states.to(self.net.device)
        dones = dones.to(self.net.device)
        weights = weights.to(self.net.device)

        # Current Q-values
        qvals = torch.gather(self.net(states), 1, actions)
        
        # Target Q-values
        with torch.no_grad():
            qvals_next = torch.max(self.target_network(next_states), dim=-1)[0].unsqueeze(1)
        
        # Bellman equation
        expected_qvals = rewards + self.gamma * qvals_next * (1 - dones)
        
        # Calculate TD errors for priority updates
        td_errors = (qvals - expected_qvals).detach().cpu().numpy().flatten()
        
        # Weighted MSE loss
        loss = (weights * (qvals - expected_qvals) ** 2).mean()
        
        return loss, (indices, td_errors)
    

    def calculate_loss_doubleDQN_prioritized(self, batch):
        states, actions, rewards, next_states, dones, weights, indices = batch
        
        states = states.to(self.net.device)
        actions = actions.to(self.net.device)
        rewards = rewards.to(self.net.device)
        next_states = next_states.to(self.net.device)
        dones = dones.to(self.net.device)
        weights = weights.to(self.net.device)
        
        # Current Q-values
        qvals = torch.gather(self.net(states), 1, actions)
    
        # Double DQN: separate action selection and evaluation
        with torch.no_grad():
            # Online network selects actions
            next_actions = torch.argmax(self.net(next_states), dim=-1, keepdim=True)
            # Target network evaluates those actions
            qvals_next = torch.gather(self.target_network(next_states), 1, next_actions)
        
        expected_qvals = rewards + self.gamma * qvals_next * (1 - dones)
        
        # Calculate TD errors for priority updates
        td_errors = (qvals - expected_qvals).detach().cpu().numpy().flatten()
        
        # Weighted MSE loss
        loss = (weights * (qvals - expected_qvals) ** 2).mean()
        
        return loss, (indices, td_errors)


    def update(self):
        # Only update if buffer has enough samples
        if len(self.buffer) < self.batch_size:
            return
            
        self.net.optimizer.zero_grad()  

        if self.use_prioritized_replay:
            # Update beta for importance sampling
            self.beta = min(1.0, self.beta_start + (1.0 - self.beta_start) * (self.step_count / self.beta_frames))
            batch = self.buffer.sample(batch_size=self.batch_size, beta=self.beta)
            
            # Calculate loss with prioritized replay
            if self.model_type == 'DQN':
                loss, priority_data = self.calculate_loss_prioritized(batch)
            elif self.model_type == 'DoubleDQN':
                loss, priority_data = self.calculate_loss_doubleDQN_prioritized(batch)
            
            # Update priorities in buffer
            if priority_data is not None:
                indices, td_errors = priority_data
                self.buffer.update_priorities(indices, td_errors)
        else:
            batch = self.buffer.sample(batch_size=self.batch_size)
            
            # Calculate loss with standard replay
            if self.model_type == 'DQN':
                loss = self.calculate_loss(batch)
            elif self.model_type == 'DoubleDQN':
                loss = self.calculate_loss_doubleDQN(batch)

        loss.backward() 
        
        # Gradient clipping for stability
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=10.0)
        
        self.net.optimizer.step() 
        self.update_loss.append(loss.item())
    

    def check_buffer_diversity(self):
        """Check if buffer has diverse experiences"""
        if len(self.buffer.buffer) < 100:
            return
        
        sample_states = [exp.state for exp in list(self.buffer.buffer)[:100]]
        sample_array = np.array(sample_states)
        
        print("\n=== BUFFER DIVERSITY CHECK ===")
        print(f"Sample states shape: {sample_array.shape}")
        print(f"State mean: {sample_array.mean():.4f}")
        print(f"State std: {sample_array.std():.4f}")
        print(f"Unique values check: {len(np.unique(sample_array[:10].flatten()))}")
        
        # Check if states are too similar (potential frame stacking bug)
        if sample_array.std() < 0.01:
            print("WARNING: States have very low variance! Check frame stacking.")
        print("=== CHECK COMPLETE ===\n")


    def save_N_checkpoints(self, episode, save_dir='checkpoints', keep_last_n=5, run_config=None):
        os.makedirs(save_dir, exist_ok=True)
        
        checkpoint = {
            'episode': episode,
            'model_state_dict': self.net.state_dict(),
            'target_model_state_dict': self.target_network.state_dict(),
            'optimizer_state_dict': self.net.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'step_count': self.step_count,
            'training_rewards': self.training_rewards,
            'mean_training_rewards': self.mean_training_rewards,
            'use_prioritized_replay': self.use_prioritized_replay,
            'beta': self.beta if self.use_prioritized_replay else None,
            'run_config': run_config,  # Store configuration
        }
        
        # Create filename with configuration
        config_str = f"{self.model_type}_{'PER' if self.use_prioritized_replay else 'ER'}"
        
        filepath = os.path.join(save_dir, f'{config_str}_ep_{episode}.pt')
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
        
        # Delete old checkpoints (only for this configuration)
        checkpoint_pattern = f'{config_str}_ep_*.pt'
        checkpoints = sorted(glob.glob(os.path.join(save_dir, checkpoint_pattern)))
        if len(checkpoints) > keep_last_n:
            for old_checkpoint in checkpoints[:-keep_last_n]:
                os.remove(old_checkpoint)
                print(f"Removed old checkpoint: {old_checkpoint}")
        
        # Save best model (specific to this configuration)
        if len(self.mean_training_rewards) > 0:
            current_mean = self.mean_training_rewards[-1]
            best_path = os.path.join(save_dir, f'{config_str}_best_ep_{episode}.pt')
            best_pattern = f'{config_str}_best_ep_*.pt'
            
            existing_best = sorted(glob.glob(os.path.join(save_dir, best_pattern)))
            
            if not existing_best:
                torch.save(checkpoint, best_path)
                print(f"Best model saved: {best_path}")
            else:
                # Load the most recent best checkpoint
                latest_best = existing_best[-1]
                best_checkpoint = torch.load(latest_best)
                best_mean = max(best_checkpoint['mean_training_rewards'][-100:])
                
                if current_mean > best_mean:
                    # Remove old best checkpoint
                    for old_best in existing_best:
                        os.remove(old_best)
                    # Save new best
                    torch.save(checkpoint, best_path)
                    print(f"New best model! (Mean: {current_mean:.2f}) - Saved: {best_path}")

                

    def load_checkpoint(self, filepath):
        #Loads a model checkpoint"""
        checkpoint = torch.load(filepath, map_location=self.net.device)
        
        self.net.load_state_dict(checkpoint['model_state_dict'])
        self.target_network.load_state_dict(checkpoint['target_model_state_dict'])
        self.net.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.step_count = checkpoint['step_count']
        self.training_rewards = checkpoint['training_rewards']
        self.mean_training_rewards = checkpoint['mean_training_rewards']

        if 'beta' in checkpoint and checkpoint['beta'] is not None:
            self.beta = checkpoint['beta']
        
        print(f"\nCheckpoint loaded from {filepath}")
        print(f"Episode: {checkpoint['episode']}")
        print(f"Steps: {self.step_count}")
        print(f"Epsilon: {self.epsilon:.3f}")
        if self.use_prioritized_replay:
            print(f"Beta: {self.beta:.3f}")
        if len(self.mean_training_rewards) > 0:
            print(f"Mean reward: {self.mean_training_rewards[-1]:.2f}\n")
        
        return checkpoint['episode']  #return episode number