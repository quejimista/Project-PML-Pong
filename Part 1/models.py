import torch
import torch.nn as nn        
import torch.optim as optim 
from torchsummary import summary
import numpy as np
import random
import os
from collections import deque

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def make_DQN(input_shape, output_shape):
    net = nn.Sequential(
        nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4),
        nn.ReLU(),
        nn.Conv2d(32, 64, kernel_size=4, stride=2),
        nn.ReLU(),
        nn.Conv2d(64, 64, kernel_size=3, stride=1),
        nn.ReLU(),
        nn.Flatten(),
        nn.Linear(64*7*7, 512),
        nn.ReLU(),
        nn.Linear(512, output_shape)
    )
    return net


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




class DoubleDQNAgent:
    def __init__(self, obs_shape, n_actions, device="cpu", seed=42,
                 replay_capacity=100000, batch_size=32, lr=1e-4, gamma=0.99,
                 target_update_freq=1000, min_replay_size=5000, eps_start=1.0,
                 eps_end=0.01, eps_decay_steps=1_000_000):
        """takes a parameters: 
        obs_shape: the shape of the observations
        n_actions: the number of actions
        device: the device to use
        seed: the seed to use
        replay_capacity: the capacity of the replay buffer
        batch_size: the batch size to use
        lr: the learning rate
        gamma: the discount factor
        target_update_freq: the frequency to update the target network
        min_replay_size: the minimum size of the replay buffer
        eps_start: the starting value of epsilon
        eps_end: the ending value of epsilon
        eps_decay_steps: the number of steps to decay epsilon
        """
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device(device)
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.min_replay_size = min_replay_size

        C, H, W = obs_shape
        self.q_net = CnnDQN(C, n_actions).to(self.device)
        self.target_net = CnnDQN(C, n_actions).to(self.device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.optimizer = optim.Adam(self.q_net.parameters(), lr=lr)

        self.replay = ReplayBuffer(replay_capacity, obs_shape, device=self.device)
        self.steps_done = 0

        # epsilon schedule
        self.eps_start = eps_start
        self.eps_end = eps_end
        self.eps_decay_steps = eps_decay_steps

    def act(self, state):
        # state: numpy (C,H,W)
        eps_threshold = self.eps_by_step()
        if random.random() < eps_threshold:
            return random.randrange(self.n_actions)
        with torch.no_grad():
            s = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            qvals = self.q_net(s)
            return int(torch.argmax(qvals, dim=1).item())

    def eps_by_step(self):
        fraction = min(float(self.steps_done) / self.eps_decay_steps, 1.0)
        return self.eps_start + fraction * (self.eps_end - self.eps_start)

    def store_transition(self, s, a, r, s_next, done):
        self.replay.add(s, a, r, s_next, done)

    def update(self):
        # check buffer
        if len(self.replay) < max(self.min_replay_size, self.batch_size):
            return None

        batch = self.replay.sample(self.batch_size)
        states = torch.tensor(batch["states"], device=self.device)  # (B,C,H,W)
        actions = torch.tensor(batch["actions"], device=self.device).long().unsqueeze(1)  # (B,1)
        rewards = torch.tensor(batch["rewards"], device=self.device).unsqueeze(1)  # (B,1)
        next_states = torch.tensor(batch["next_states"], device=self.device)  # (B,C,H,W)
        dones = torch.tensor(batch["dones"].astype(np.uint8), device=self.device).unsqueeze(1)  # (B,1)

        # Q(s,a) for actions taken
        q_values = self.q_net(states).gather(1, actions)  # (B,1)

        # Double DQN target:
        # a* = argmax_a Q_main(s', a)
        with torch.no_grad():
            next_q_main = self.q_net(next_states)  # (B, n_actions)
            next_actions = next_q_main.argmax(dim=1, keepdim=True)  # (B,1)
            next_q_target = self.target_net(next_states)  # (B, n_actions)
            next_q_target_values = next_q_target.gather(1, next_actions)  # (B,1)
            target = rewards + self.gamma * (1 - dones.float()) * next_q_target_values

        loss = nn.functional.mse_loss(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        # gradient clipping (helpful)
        nn.utils.clip_grad_norm_(self.q_net.parameters(), 10.0)
        self.optimizer.step()

        # update steps and target network copy
        self.steps_done += 1
        if self.steps_done % self.target_update_freq == 0:
            self.target_net.load_state_dict(self.q_net.state_dict())

        return loss.item()

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            'q_state_dict': self.q_net.state_dict(),
            'target_state_dict': self.target_net.state_dict(),
            'optimizer_state': self.optimizer.state_dict()
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.q_net.load_state_dict(data['q_state_dict'])
        self.target_net.load_state_dict(data['target_state_dict'])
        self.optimizer.load_state_dict(data['optimizer_state'])