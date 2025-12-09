import argparse
import time
import gymnasium as gym
import numpy as np
import torch
import imageio
from stable_baselines3 import PPO
# We import your specific preprocessing pipeline
from functions.preprocessing_aina import make_env

def load_and_watch(model_path, delay=0.02):
    """
    Loads the model and renders the gameplay in a window (Human mode).
    Use this to visually inspect if the agent is trying to hit gates.
    """
    print(f"\nLOADING MODEL: {model_path}")
    
    # 1. Load the Model
    # We explicitly map to CPU to avoid CUDA version errors during simple visualization
    model = PPO.load(model_path, device="cpu")
    
    # 2. Create the Environment
    # We assume norm_obs=False (Raw Pixels) as per the latest fix.
    # If the agent acts crazily/randomly, it might mean it expects normalized obs 
    # but we are giving it raw 0-255 pixels.
    env = make_env("ALE/Skiing-v5", render="human", verbose=False)
    
    print("\nSTARTING PLAYBACK...")
    print("Press Ctrl+C to stop.")
    print("-" * 50)
    
    try:
        obs, _ = env.reset()
        episode_reward = 0
        steps = 0
        
        while True:
            # Predict action (Deterministic=True means no randomness, purely the learned policy)
            action, _ = model.predict(obs, deterministic=True)
            
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            steps += 1
            
            # Artificial delay so you can actually see what's happening
            time.sleep(delay)
            
            if terminated or truncated:
                print(f"Episode Finished! Score: {episode_reward:.2f} | Steps: {steps}")
                
                # Check for "The Suicide Strategy"
                if steps < 500:
                    print(">>> DIAGNOSIS: Agent crashed/quit early (The Suicide Strategy).")
                elif episode_reward > -500 and episode_reward < -200:
                    print(">>> DIAGNOSIS: Agent likely went straight down (The Local Minimum).")
                
                obs, _ = env.reset()
                episode_reward = 0
                steps = 0
                time.sleep(1.0)
                
    except KeyboardInterrupt:
        print("\nPlayback stopped by user.")
    finally:
        env.close()


def record_video(model_path, video_filename="skiing_agent.mp4", max_steps=5000):
    """
    Records a full episode and saves it as an MP4 file.
    This is useful for the project deliverable.
    """
    print(f"\nRECORDING VIDEO TO: {video_filename}")
    
    model = PPO.load(model_path, device="cpu")
    
    # We use 'rgb_array' to get the pixel data for the video
    env = make_env("ALE/Skiing-v5", render="rgb_array", verbose=False)
    
    obs, _ = env.reset()
    done = False
    steps = 0
    images = []
    
    try:
        while not done and steps < max_steps:
            # Capture the frame
            # Note: We need to grab the original frame, not the preprocessed grayscale stack
            # Gymnasium's render() gives us the game screen.
            frame = env.render() 
            images.append(frame)
            
            action, _ = model.predict(obs, deterministic=True)
            obs, _, terminated, truncated, _ = env.step(action)
            
            done = terminated or truncated
            steps += 1
            
            if steps % 100 == 0:
                print(f"Recorded {steps} steps...")
                
    except KeyboardInterrupt:
        print("Recording interrupted...")
    finally:
        env.close()
        
    if len(images) > 0:
        print(f"Saving video with {len(images)} frames...")
        imageio.mimsave(video_filename, images, fps=30)
        print("Done!")
    else:
        print("No frames captured.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize Skiing Agent")
    parser.add_argument("--model", type=str, required=True, help="Path to .zip file")
    parser.add_argument("--record", action="store_true", help="Record .mp4 instead of watching")
    parser.add_argument("--out", type=str, default="skiing_agent.mp4", help="Output filename for recording")
    
    args = parser.parse_args()
    
    if args.record:
        record_video(args.model, args.out)
    else:
        load_and_watch(args.model)