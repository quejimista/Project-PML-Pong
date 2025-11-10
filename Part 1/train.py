#!/usr/bin/env python3
import gymnasium as gym
from lib.DQN import *
from lib.wrappers import *
from lib.Agent import *
from lib.ExperienceBuffer import *

import argparse
import time
import numpy as np
import typing as tt
import datetime
import os
import wandb

import torch
import torch.nn as nn
import torch.optim as optim


### Parameters
DEFAULT_ENV_NAME = "PongNoFrameskip-v4"
MEAN_REWARD_BOUND = 19

GAMMA = 0.99
BATCH_SIZE = 32
REPLAY_SIZE = 10000
LEARNING_RATE = 1e-4
SYNC_TARGET_FRAMES = 1000
REPLAY_START_SIZE = 10000

EPSILON_DECAY_LAST_FRAME = 150000
EPSILON_START = 1.0
EPSILON_FINAL = 0.01


def batch_to_tensors(batch: tt.List[Experience], device: torch.device) -> BatchTensors:
    """
    Convert a batch of Experience to a tuple of tensors.

    Args:
        batch (tt.List[Experience]): A list of Experience objects.
        device (torch.device): The device to which the tensors will be moved.

    Returns:
        BatchTensors: A tuple containing the tensors for states, actions, rewards, dones, and new_states.
    """
    states, actions, rewards, dones, new_state = [], [], [], [], []
    for e in batch:
        states.append(e.state)
        actions.append(e.action)
        rewards.append(e.reward)
        dones.append(e.done_trunc)
        new_state.append(e.new_state)
    states_t = torch.as_tensor(np.asarray(states))
    actions_t = torch.LongTensor(actions)
    rewards_t = torch.FloatTensor(rewards)
    dones_t = torch.BoolTensor(dones)
    new_states_t = torch.as_tensor(np.asarray(new_state))

    return states_t.to(device), actions_t.to(device), rewards_t.to(device), dones_t.to(device),  new_states_t.to(device)


def calc_loss(batch: tt.List[Experience], net: DQN, tgt_net: DQN, device: torch.device) -> torch.Tensor:
    """
    Calculate the loss for a batch of experiences.
    
    Args:
        batch (tt.List[Experience]): A list of Experience objects.
        net (DQN): The current DQN network.
        tgt_net (DQN): The target DQN network.
        device (torch.device): The device to which the tensors will be moved.
    
    Returns:
        torch.Tensor: The calculated loss.
    """
    states_t, actions_t, rewards_t, dones_t, new_states_t = batch_to_tensors(batch, device)

    state_action_values = net(states_t).gather(1, actions_t.unsqueeze(-1)).squeeze(-1)
    with torch.no_grad():
        next_state_values = tgt_net(new_states_t).max(1)[0]
        next_state_values[dones_t] = 0.0
        next_state_values = next_state_values.detach()

    expected_state_action_values = next_state_values * GAMMA + rewards_t
    
    return nn.MSELoss()(state_action_values, expected_state_action_values)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", default="cpu", help="Device name, default=cpu")
    parser.add_argument("--env", default=DEFAULT_ENV_NAME, help="Name of the environment, default=" + DEFAULT_ENV_NAME)
    args = parser.parse_args()
    device = torch.device(args.dev)

    # Gymnasium version
    print("Using Gymnasium version {}".format(gym.__version__))
    # WanDB login
    wandb.login()
    # start a new wandb run to track this script
    wandb.init(project="M3-2_Example_3")

    env = make_env(args.env)
    net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    tgt_net = DQN(env.observation_space.shape, env.action_space.n).to(device)
    print(net)

    buffer = ExperienceBuffer(REPLAY_SIZE)
    agent = Agent(env, buffer)
    epsilon = EPSILON_START

    optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
    total_rewards = []
    frame_idx = 0
    ts_frame = 0
    ts = time.time()
    best_m_reward = None

    print(">>> Training starts at ",datetime.datetime.now())

    while True:
        frame_idx += 1
        epsilon = max(EPSILON_FINAL, EPSILON_START - frame_idx / EPSILON_DECAY_LAST_FRAME)

        reward = agent.play_step(net, device, epsilon)
        if reward is not None:
            total_rewards.append(reward)
            speed = (frame_idx - ts_frame) / (time.time() - ts)
            ts_frame = frame_idx
            ts = time.time()
            m_reward = np.mean(total_rewards[-100:])
            print(f"{frame_idx}: done {len(total_rewards)} games, reward {m_reward:.3f}, eps {epsilon:.2f}, speed {speed:.2f} f/s")
            wandb.log({"epsilon": epsilon, "speed": speed, "reward_100": m_reward, "reward": reward}, step=frame_idx)

            if best_m_reward is None or best_m_reward < m_reward:
                if best_m_reward is not None:
                    print(f"Best reward updated {best_m_reward:.3f} -> {m_reward:.3f}")
                    model_name = os.path.join("models", args.env + "_DQN.pth")
                    print(f"Saving model '{model_name}'")
                    torch.save(net.state_dict(), model_name)
                best_m_reward = m_reward
            if m_reward > MEAN_REWARD_BOUND:
                print("Solved in %d frames!" % frame_idx)
                break
        if len(buffer) < REPLAY_START_SIZE:
            continue
        if frame_idx % SYNC_TARGET_FRAMES == 0:
            tgt_net.load_state_dict(net.state_dict())

        optimizer.zero_grad()
        batch = buffer.sample(BATCH_SIZE)
        loss_t = calc_loss(batch, net, tgt_net, device)
        loss_t.backward()
        optimizer.step()

    print(">>> Training ends at ",datetime.datetime.now())
    wandb.finish()