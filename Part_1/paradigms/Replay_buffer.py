import torch
import random
import numpy as np
import collections

# queue
Experience = collections.namedtuple('Experience', field_names=['state', 'action', 'reward', 'done', 'new_state'])

class ReplayBuffer:
    def __init__(self, capacity=50000, burn_in=10000):
        self.buffer = collections.deque(maxlen=capacity) #when max capacity, pop the oldest
        self.burn_in = burn_in
        self.capacity = capacity

    def append(self, experience): #saves experience as tuple in buffer
        self.buffer.append(experience) #(S, A, R, S', D)

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, dones, next_states = zip(*batch)

        # Convert lists → numpy arrays
        states = np.array(states)
        next_states = np.array(next_states)

        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1)
        )

    def __len__(self):
        return len(self.buffer) #number of elements in the buffer, useful to know if there are enough to train
    
    # The buffer is filled with random experiences at the beginning of training
    def burn_in_capacity(self):
        return len(self.buffer) / self.burn_in
    

#-------------------------------

class PrioritizedReplayBuffer:
    def __init__(self, capacity=50000, burn_in=10000, alpha=0.6):
        self.capacity = capacity
        self.burn_in = burn_in
        self.alpha = alpha
        #prioritization parameter [0, 1]
        #how much prioritization is used (0 = uniform (normal buffer),
        #1 = full (samples with more TD error are more likely to be sampled))
        self.buffer = [] #list to store experiences (s, a, r, s', done)
        self.priorities = np.zeros((capacity,), dtype=np.float32) #array to store priorities, init to 0
        self.pos = 0  #pointer to the next insert position

    def append(self,experience):
        max_prio = self.priorities.max() if self.buffer else np.float32(1.0) #if buffer is empty, max_prio = 1

        if len(self.buffer) < self.capacity: # if buffer is not full, append the experience
            self.buffer.append(experience)
        else:
            self.buffer[self.pos] = experience # if its full: replace the experience

        self.priorities[self.pos] = max_prio # new priority takes the max priority
        self.pos = (self.pos + 1) % self.capacity # move pointer to next position

    def sample(self, batch_size, beta=0.4):
        if len(self.buffer) == self.capacity: # if buffer is full, use all priorities
            prios = self.priorities
        else: # if not full, only use priorities up to the last inserted element.
            prios = self.priorities[:len(self.buffer)]

        # calculate sampling probabilities p(i) from priorities p_i: p(i) = p_i^alpha / sum(p_j^alpha)
        probs = prios ** self.alpha
        probs /= probs.sum()

        # sample the batch indices based on the calculated probabilities (weighted sampling)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        experiences = [self.buffer[i] for i in indices] # get the experiences with the indices

        # ----------compute importance-sampling weights---------- 
        # correct the bias introduced by non uniform sampling
        total = len(self.buffer) #size of the buffer
        weights = (total * probs[indices]) ** (-beta)   # in the slides is 1/N * 1/p_i ^ beta, 
                                                        # but this should be equivalent and easier to compute this way
        weights /= weights.max()  # normalize to 1

        # Extract components from Experience namedtuples
        states = np.array([exp.state for exp in experiences])
        actions = np.array([exp.action for exp in experiences])
        rewards = np.array([exp.reward for exp in experiences])
        next_states = np.array([exp.new_state for exp in experiences])
        dones = np.array([exp.done for exp in experiences], dtype=np.float32)

        # Convert to torch tensors
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1),
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1),
            torch.tensor(weights, dtype=torch.float32),
            indices  # Return indices for priority updates
        )

    def update_priorities(self, indices, td_errors, epsilon=1e-6):
        for idx, td in zip(indices, td_errors):
            # New priority is the absolute td error plus a small epsilon
                    # Large TD error then it is sampled more often (important transitions)
            # Epsilon prevents priorities from being zero, ensuring a minimum sampling probability
            self.priorities[idx] = abs(td) + epsilon

    def burn_in_capacity(self):
        # Check if buffer has enough samples for training
        return len(self.buffer) / self.burn_in

    def __len__(self):
        return len(self.buffer)