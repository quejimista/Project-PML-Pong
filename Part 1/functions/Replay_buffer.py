
import torch
from collections import deque
import random
import numpy as np

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity) #when max capacity, pop the oldest

    def push(self, state, action, reward, next_state, done): #saves experience as tuple in buffer
        self.buffer.append((state, action, reward, next_state, done)) #(S, A, R, S', D)

    def sample(self, batch_size): #random sampling from the buffer, to break correlation
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch) #states = [s1, s2, ...], actions = [a1, a2, ...], etc.
        #result is a tuple like this: batch = 3: 
        # [(s1, a1, r1, s1_next, d1), (s2, a2, r2, s2_next, d2),(s3, a3, r3, s3_next, d3)], etc.
        
        return (
            torch.tensor(np.array(states), dtype=torch.float32), #current state
            torch.tensor(actions, dtype=torch.int64).unsqueeze(1), #actions (column vector)
            torch.tensor(rewards, dtype=torch.float32).unsqueeze(1), #rewards (column vector)
            torch.tensor(np.array(next_states), dtype=torch.float32), #next states
            torch.tensor(dones, dtype=torch.float32).unsqueeze(1) #episode is done or not flags.
        )

    def __len__(self):
        return len(self.buffer) #number of elements in the buffer, useful to know if there are enough to train