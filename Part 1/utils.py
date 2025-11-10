import numpy as np
import torch
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