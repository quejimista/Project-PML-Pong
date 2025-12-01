#!/usr/bin/env python3
import gymnasium as gym
import numpy as np
import torch
import os
from functions.preprocessing import make_env
from functions.models import DQN


MODEL_PATH = r"C:\Users\ainav\OneDrive\Documents\Uni\4th_year\1st_semester\paradigms_ml\project\Project-PML-Pong\Part_1\download_models_wandb\DoubleDQN_PER_lor0.0001_dnnupd4_epsdec0.99_batch64_20251139_124235.pt"         # ← Just change this
# MODEL_PATH = r'C:\Users\ainav\OneDrive\Documents\Uni\4th_year\1st_semester\paradigms_ml\project\Project-PML-Pong\Part_1\download_models_wandb\DQN_PER_lr0.0001_dnnupd4_epsdec0.99_batch32_20251130_094810.pt'
# MODEL_PATH = r'C:\Users\ainav\OneDrive\Documents\Uni\4th_year\1st_semester\paradigms_ml\project\Project-PML-Pong\Part_1\download_models_wandb\DoubleDQN_PER_lr0.0001_dnnupd4_epsdec0.99_batch32_20251129_195152.pt'

MODEL_NAME = f"DoubleDQN_PER_lor0.0001_dnnupd4_epsdec0.99_batch64_20251139_124235"
# MODEL_NAME = f'DQN_PER_lr0.0001_dnnupd4_epsdec0.99_batch32_20251130_094810'
# MODEL_NAME = "DoubleDQN_PER_lr0.0001_dnnupd4_epsdec0.99_batch32_20251129_195152"


ENV_NAME = "PongNoFrameskip-v4"
VIDEO_FOLDER = f"test_video/{MODEL_NAME}"             # Where to save video


if __name__ == "__main__":
    os.makedirs(VIDEO_FOLDER, exist_ok=True)

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    # Create environment
    env = make_env(ENV_NAME, render_mode="rgb_array")

    # Add video recorder
    env = gym.wrappers.RecordVideo(
        env,
        video_folder=VIDEO_FOLDER,
        episode_trigger=lambda ep: True
    )

    # Create network
    net = DQN(env, device=device)
    net.eval()

    # Load weights
    state_dict = torch.load(MODEL_PATH, map_location=device)
    net.load_state_dict(state_dict)
    print(f"\nLoaded model: {MODEL_PATH}")

    # Run evaluation
    state, _ = env.reset()
    total_reward = 0
    action_counts = {}

    done = False
    while not done:
        t_state = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            qvals = net(t_state)
        action = int(torch.argmax(qvals, dim=1).item())

        action_counts[action] = action_counts.get(action, 0) + 1

        next_state, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        done = terminated or truncated
        state = next_state

    env.close()

    print("\n===== TEST RESULTS =====")
    print(f"Total reward: {total_reward}")
    print(f"Action counts: {action_counts}")
    print(f"Video saved in: {VIDEO_FOLDER}")
