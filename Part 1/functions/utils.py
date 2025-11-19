import numpy as np
import torch
import matplotlib.pyplot as plt
import numpy as np
#------------------------------------epsilon schedulers------------------------------------
#base class for epsilon schedulers
class EpsilonScheduler:
    def __init__(self, start, end):
        self.start = start
        self.end = end
        self.epsilon = start

    def step(self):
        raise NotImplementedError #update epsilon

    def get(self):
        return self.epsilon #get epsilon

class LinearDecay(EpsilonScheduler):
    def __init__(self, start, end, decay_steps):
        super().__init__(start, end)
        self.decay_steps = decay_steps
        self.step_count = 0

    def step(self):
        self.step_count += 1
        #linear decay
        self.epsilon = max(
            self.end,
            self.start - (self.start - self.end) * (self.step_count / self.decay_steps)
        )
        return self.epsilon
    
class ExponentialDecay(EpsilonScheduler):
    def __init__(self, start, end, decay_rate):
        super().__init__(start, end)
        self.decay_rate = decay_rate

    def step(self):
        #exponential decay
        self.epsilon = max(self.end, self.epsilon * self.decay_rate) #get the higher value between the minimum or the next epsilon
        return self.epsilon #get epsilon after exponential decay
#-------------------------------------------------------------------------------------------

def epsilon_soft_action(q_net, state, n_actions, epsilon, device):
    state = torch.tensor(np.array([state]), dtype=torch.float32).to(device)
    q_values = q_net(state).detach().cpu().numpy().flatten() #get q-values as 1d array

    a_star = np.argmax(q_values) #best action
    probs = np.ones(n_actions) * (epsilon / n_actions) #initialize probabilities
    probs[a_star] += 1.0 - epsilon #increase probability for best action

    action = np.random.choice(np.arange(n_actions), p=probs)#sample action
    return int(action) #return action





# --------------------------------------visualization------------------------------------------------------
def plot_training_results(agent, save_path="training_plot.png"):
    episodes = np.arange(len(agent.training_rewards))
    
    # Create a figure with 3 subplots
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Rewards Plot
    ax1.plot(episodes, agent.training_rewards, label='Episode Reward', color='lightalpha_blue', alpha=0.3)
    ax1.plot(episodes, agent.mean_training_rewards, label='100-Ep Moving Avg', color='blue')
    ax1.axhline(y=agent.reward_threshold, color='r', linestyle='--', label='Solved Threshold')
    ax1.set_ylabel('Reward')
    ax1.set_title('Training Rewards (Pong)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Loss Plot
    # Handle cases where loss might be empty initially or have different length
    loss_len = len(agent.training_loss_history)
    ax2.plot(np.arange(loss_len), agent.training_loss_history, label='Avg Episode Loss', color='orange')
    ax2.set_ylabel('MSE Loss')
    ax2.set_title('Training Loss')
    ax2.set_yscale('log') # Log scale often helps visualize Loss better
    ax2.grid(True, alpha=0.3)
    
    # 3. Epsilon Decay Plot
    eps_len = len(agent.epsilon_history)
    ax3.plot(np.arange(eps_len), agent.epsilon_history, label='Epsilon', color='green')
    ax3.set_ylabel('Epsilon')
    ax3.set_xlabel('Episode')
    ax3.set_title('Exploration Decay')
    ax3.grid(True, alpha=0.3)

    plt.tight_layout()
    
    # Save and Close
    plt.savefig(save_path)
    print(f"\nPlot saved to {save_path}")
    plt.close()

