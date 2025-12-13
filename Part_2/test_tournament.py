import argparse
import sys
import numpy as np
import supersuit as ss
from pettingzoo.atari import pong_v3
from tournament_agent import TournamentAgent

def make_env(render_mode=None):
    """
    Creates the PettingZoo Pong environment with tournament preprocessing.
    """
    env = pong_v3.env(num_players=2, render_mode=render_mode)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    return env

def run_tournament_test(model_path, n_episodes=5, render=False):
    """
    Runs a self-play tournament match with LIVE terminal updates.
    """
    print(f"\n{'='*70}")
    print(f"STARTING SELF-PLAY TOURNAMENT TEST")
    print(f"{'='*70}")
    print(f"Mode: {'Visual Render' if render else 'Headless/Server'}")
    print(f"Agents: Left (Mirror) vs Right (Standard)")
    print(f"{'='*70}\n")
    
    # 1. Initialize Agents
    try:
        agent_left = TournamentAgent(model_path)
        agent_right = TournamentAgent(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    # 2. Setup Environment
    render_mode = "human" if render else None
    env = make_env(render_mode=render_mode)

    # 3. Run Episodes
    left_wins = 0
    right_wins = 0
    
    for episode in range(n_episodes):
        env.reset()
        scores = {agent: 0 for agent in env.possible_agents}
        steps = 0
        
        print(f"Episode {episode+1}/{n_episodes} Started...")
        
        # Initial status line
        sys.stdout.write(f"\r   Step {steps:04d} | Score: Left 0 - 0 Right")
        sys.stdout.flush()

        for agent_id in env.agent_iter():
            observation, reward, termination, truncation, info = env.last()
            
            # Accumulate rewards
            if agent_id in scores:
                scores[agent_id] += reward
                
                # If a point was scored (reward is non-zero), update the visual log
                # Note: In Pong, reward is +1 or -1. We check the accumulated score.
                # However, usually reward is only non-zero when a point ends.
                if reward != 0:
                    # Update the live status line
                    p1 = int(scores["second_0"]) # Left
                    p2 = int(scores["first_0"])  # Right
                    sys.stdout.write(f"\r   Step {steps:04d} | Score: Left {p1} - {p2} Right")
                    sys.stdout.flush()

            if termination or truncation:
                action = None
            else:
                if agent_id == "first_0":
                    action = agent_right.predict(observation, side="right")
                    steps += 1 # Count steps (roughly, since 2 agents move)
                else:
                    action = agent_left.predict(observation, side="left")
            
            env.step(action)
        
        # End of episode summary
        p1_final = int(scores["second_0"]) # Left
        p2_final = int(scores["first_0"])  # Right
        
        if p2_final > p1_final:
            right_wins += 1
            winner = "RIGHT"
        elif p1_final > p2_final:
            left_wins += 1
            winner = "LEFT"
        else:
            winner = "DRAW"
            
        # Print final line for this episode (overwriting the last status line)
        sys.stdout.write(f"\r   Step {steps:04d} | Final: Left {p1_final} - {p2_final} Right | Winner: {winner}\n")
        sys.stdout.flush()

    env.close()
    
    print(f"\n{'='*70}")
    print(f"TOURNAMENT RESULTS")
    print(f"{'='*70}")
    print(f"Left Agent (Mirrored) Wins:  {left_wins}  ({(left_wins/n_episodes)*100:.1f}%)")
    print(f"Right Agent (Standard) Wins: {right_wins}  ({(right_wins/n_episodes)*100:.1f}%)")
    print(f"{'='*70}\n")

if __name__ == "__main__":
    MODEL_PATH = r'C:\Users\ainav\OneDrive\Documents\Uni\4th_year\1st_semester\paradigms_ml\project\Project-PML-Pong\Part_2\exports\pong\best_model\best_model.zip'
    
    # Run 5 episodes of self-play
    run_tournament_test(MODEL_PATH, n_episodes=5, render=False)