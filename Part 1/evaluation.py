"""
Script to evaluate and visualize a trained DQN agent playing Pong
"""

import gymnasium as gym
import torch
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import time
import os
import ale_py
# Import your agent and network classes
from functions.Agent import Agent
from functions.models import *  # Your DQN network
from functions.Replay_buffer import ReplayBuffer
from functions.preprocessing import make_env

class AgentEvaluator:
    def __init__(self, checkpoint_path, env_name='PongNoFrameskip-v4', render_mode='rgb_array'):
        """
        Initialize the evaluator
        
        Args:
            checkpoint_path: Path to the checkpoint file
            env_name: Gymnasium environment name
            render_mode: 'human' for live display, 'rgb_array' for recording
        """
        self.checkpoint_path = checkpoint_path
        self.env_name = env_name
        self.render_mode = render_mode
        
        # Create environment
        gym.register_envs(ale_py)
        self.env = self._create_preprocessed_env(env_name, render_mode)
        
        
        # Create a dummy agent to load the checkpoint
        # You'll need to adjust this based on your network architecture
        self.agent = self._create_agent()
        
        # Load the checkpoint
        self.load_checkpoint()
        
        print(f"Agent loaded from {checkpoint_path}")
        print(f"   Environment: {env_name}")
        print(f"   Render mode: {render_mode}")
    
    def _create_preprocessed_env(self, env_name, render_mode):
        """Create environment with same preprocessing as training"""
        from functions.preprocessing import make_env
        return make_env(env_name, render=render_mode)

    def _create_agent(self):
        """Create agent with same architecture as training"""
        # Detect device
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        
        # Create network - pass the environment
        net = DQN(self.env, learning_rate=1e-3, device=device)  # <-- CAMBIO AQUÍ
        
        # Create buffer (not really needed for evaluation but required by Agent)
        buffer = ReplayBuffer(capacity=1000)
        
        # Create agent
        agent = Agent(
            env=self.env,
            net=net,
            buffer=buffer,
            epsilon=0.0,  # No exploration during evaluation
        )
    
        return agent
    
    def load_checkpoint(self):
        """Load weights from checkpoint"""
        checkpoint = torch.load(self.checkpoint_path, 
                              map_location=self.agent.net.device,
                              weights_only=False)
        
        self.agent.net.load_state_dict(checkpoint['model_state_dict'])
        self.agent.net.eval()  # Set to evaluation mode
        
        # Print checkpoint info
        print(f"\nCheckpoint Info:")
        print(f"   Episode: {checkpoint['episode']}")
        print(f"   Steps: {checkpoint['step_count']}")
        print(f"   Epsilon: {checkpoint['epsilon']:.3f}")
        if len(checkpoint['mean_training_rewards']) > 0:
            print(f"   Mean Reward: {checkpoint['mean_training_rewards'][-1]:.2f}")
    
    def play_episode(self, render=True, verbose=True):
        """
        Play one episode and return statistics
        
        Returns:
            dict with episode statistics
        """
        state = self.env.reset()[0]
        total_reward = 0
        steps = 0
        done = False
        
        frames = []  # Store frames for video/gif
        actions_taken = []
        rewards_received = []
        
        while not done:
            # Get action from agent (no exploration)
            with torch.no_grad():
                state_v = torch.tensor(np.array(state, copy=False)).to(self.agent.net.device).unsqueeze(0)
                q_vals = self.agent.net(state_v)
                action = int(torch.argmax(q_vals, dim=1).item())
            
            # Take action
            next_state, reward, terminated, truncated, _ = self.env.step(action)
            done = terminated or truncated
            
            # Store info
            total_reward += reward
            steps += 1
            actions_taken.append(action)
            rewards_received.append(reward)
            
            # Capture frame if rendering
            if render and self.render_mode == 'rgb_array':
                frames.append(self.env.render())
            
            state = next_state
            
            if verbose and steps % 100 == 0:
                print(f"   Step {steps} | Reward: {total_reward:.1f}")
        
        stats = {
            'total_reward': total_reward,
            'steps': steps,
            'actions': actions_taken,
            'rewards': rewards_received,
            'frames': frames
        }
        
        return stats
    
    def evaluate(self, n_episodes=10, render=False, verbose=True):
        """
        Evaluate agent over multiple episodes
        
        Args:
            n_episodes: Number of episodes to run
            render: Whether to render (slows down evaluation)
            verbose: Print progress
        
        Returns:
            dict with evaluation statistics
        """
        print(f"\n🎮 Evaluating agent over {n_episodes} episodes...")
        
        all_rewards = []
        all_steps = []
        episode_stats = []
        
        for ep in range(n_episodes):
            if verbose:
                print(f"\nEpisode {ep + 1}/{n_episodes}")
            
            stats = self.play_episode(render=render, verbose=verbose)
            
            all_rewards.append(stats['total_reward'])
            all_steps.append(stats['steps'])
            episode_stats.append(stats)
            
            if verbose:
                print(f"Reward: {stats['total_reward']:.1f} | Steps: {stats['steps']}")
        
        # Calculate summary statistics
        results = {
            'mean_reward': np.mean(all_rewards),
            'std_reward': np.std(all_rewards),
            'min_reward': np.min(all_rewards),
            'max_reward': np.max(all_rewards),
            'mean_steps': np.mean(all_steps),
            'all_rewards': all_rewards,
            'all_steps': all_steps,
            'episode_stats': episode_stats
        }
        
        # Print summary
        print(f"\n{'='*60}")
        print(f"EVALUATION SUMMARY")
        print(f"{'='*60}")
        print(f"Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"Min/Max:     {results['min_reward']:.1f} / {results['max_reward']:.1f}")
        print(f"Mean Steps:  {results['mean_steps']:.1f}")
        print(f"{'='*60}\n")
        
        return results
    
    def plot_evaluation(self, results, save_path='evaluation_plot.png'):
        """Plot evaluation results"""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        episodes = np.arange(len(results['all_rewards']))
        
        # 1. Episode rewards
        ax = axes[0, 0]
        ax.plot(episodes, results['all_rewards'], 'o-', label='Episode Reward')
        ax.axhline(y=results['mean_reward'], color='r', linestyle='--', 
                   label=f'Mean: {results["mean_reward"]:.2f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Reward')
        ax.set_title('Evaluation Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 2. Reward distribution
        ax = axes[0, 1]
        ax.hist(results['all_rewards'], bins=20, edgecolor='black', alpha=0.7)
        ax.axvline(x=results['mean_reward'], color='r', linestyle='--', 
                   label=f'Mean: {results["mean_reward"]:.2f}')
        ax.set_xlabel('Reward')
        ax.set_ylabel('Frequency')
        ax.set_title('Reward Distribution')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Episode lengths
        ax = axes[1, 0]
        ax.plot(episodes, results['all_steps'], 'o-', color='green')
        ax.axhline(y=results['mean_steps'], color='r', linestyle='--',
                   label=f'Mean: {results["mean_steps"]:.1f}')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Steps')
        ax.set_title('Episode Lengths')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 4. Action distribution (from first episode)
        ax = axes[1, 1]
        actions = results['episode_stats'][0]['actions']
        unique, counts = np.unique(actions, return_counts=True)
        ax.bar(unique, counts)
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Distribution (First Episode)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        print(f"Plot saved to {save_path}")
        plt.close()
    
    def watch_agent_play(self, n_episodes=3):
        """Watch the agent play with live rendering"""
        print(f"\nWatching agent play {n_episodes} episodes...")
        print("   Close the window to continue to next episode\n")
        
        # Create environment with human rendering
        env = gym.make(self.env_name, render_mode='human')
        
        for ep in range(n_episodes):
            print(f"Episode {ep + 1}/{n_episodes}")
            state = env.reset()[0]
            total_reward = 0
            done = False
            
            while not done:
                # Get action
                with torch.no_grad():
                    state_v = torch.tensor(np.array(state, copy=False)).to(self.agent.net.device).unsqueeze(0)
                    q_vals = self.agent.net(state_v)
                    action = int(torch.argmax(q_vals, dim=1).item())
                
                state, reward, terminated, truncated, _ = env.step(action)
                done = terminated or truncated
                total_reward += reward
                
                time.sleep(0.01)  # Slow down a bit for viewing
            
            print(f"   Episode {ep + 1} finished with reward: {total_reward:.1f}\n")
        
        env.close()
    
    def analyze_q_values(self, n_steps=1000):
        """Analyze Q-value distributions during play"""
        print(f"\n🔍 Analyzing Q-values over {n_steps} steps...")
        
        state = self.env.reset()[0]
        q_values_all = []
        q_values_max = []
        q_values_selected = []
        actions_taken = []
        
        for step in range(n_steps):
            with torch.no_grad():
                state_v = torch.tensor(np.array(state, copy=False)).to(self.agent.net.device).unsqueeze(0)
                q_vals = self.agent.net(state_v)
                q_vals_np = q_vals.cpu().numpy()[0]
                
                action = int(torch.argmax(q_vals, dim=1).item())
                
                q_values_all.append(q_vals_np)
                q_values_max.append(np.max(q_vals_np))
                q_values_selected.append(q_vals_np[action])
                actions_taken.append(action)
            
            state, _, terminated, truncated, _ = self.env.step(action)
            if terminated or truncated:
                state = self.env.reset()[0]
        
        # Plot Q-value analysis
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # 1. Max Q-value over time
        ax = axes[0, 0]
        ax.plot(q_values_max, alpha=0.6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Max Q-value')
        ax.set_title('Max Q-value Over Time')
        ax.grid(True, alpha=0.3)
        
        # 2. Selected Q-value vs Max Q-value
        ax = axes[0, 1]
        ax.plot(q_values_max, label='Max Q-value', alpha=0.6)
        ax.plot(q_values_selected, label='Selected Q-value', alpha=0.6)
        ax.set_xlabel('Step')
        ax.set_ylabel('Q-value')
        ax.set_title('Selected vs Max Q-values')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # 3. Q-value distribution
        ax = axes[1, 0]
        all_q = np.concatenate(q_values_all)
        ax.hist(all_q, bins=50, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Q-value')
        ax.set_ylabel('Frequency')
        ax.set_title('Q-value Distribution')
        ax.grid(True, alpha=0.3)
        
        # 4. Action distribution
        ax = axes[1, 1]
        unique, counts = np.unique(actions_taken, return_counts=True)
        ax.bar(unique, counts)
        ax.set_xlabel('Action')
        ax.set_ylabel('Frequency')
        ax.set_title('Action Distribution')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('qvalue_analysis.png', dpi=150)
        print(f"Q-value analysis saved to qvalue_analysis.png")
        plt.close()
        
        # Print statistics
        print(f"\n Q-value Statistics:")
        print(f"   Mean Q-value: {np.mean(all_q):.3f}")
        print(f"   Std Q-value:  {np.std(all_q):.3f}")
        print(f"   Min Q-value:  {np.min(all_q)}")



#main                                  
CHECKPOINT_PATH = 'checkpoints/best_model.pt'  # Change this
ENV_NAME = 'PongNoFrameskip-v4'
N_EVAL_EPISODES = 10

# Create evaluator
evaluator = AgentEvaluator(
    checkpoint_path=CHECKPOINT_PATH,
    env_name=ENV_NAME,
    render_mode='human'  # Use 'human' to watch live
)

# Option 1: Full evaluation
results = evaluator.evaluate(n_episodes=N_EVAL_EPISODES, render=False, verbose=True)
# evaluator.plot_evaluation(results)

# Option 2: Watch agent play (uncomment to use)
evaluator.watch_agent_play(n_episodes=3)

# Option 3: Analyze Q-values (uncomment to use)
# evaluator.analyze_q_values(n_steps=1000)

# Close
evaluator.close()

print("\n✅ Evaluation complete!")