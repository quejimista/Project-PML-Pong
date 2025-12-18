import time
import numpy as np
import supersuit as ss
from pettingzoo.atari import pong_v3
from stable_baselines3 import PPO

# --- 1. SETUP ENVIRONMENT ---
def make_viz_env(render_mode="human"):
    """
    Creates the PettingZoo environment with the EXACT tournament preprocessing.
    """
    env = pong_v3.env(num_players=2, render_mode=render_mode)
    env = ss.color_reduction_v0(env, mode="B")
    env = ss.resize_v1(env, x_size=84, y_size=84)
    env = ss.frame_stack_v1(env, 4, stack_dim=0)
    env = ss.dtype_v0(env, dtype=np.float32)
    env = ss.normalize_obs_v0(env, env_min=0, env_max=1)
    env = ss.reshape_v0(env, (4, 84, 84))
    return env

# --- 2. DEFINE THE AGENT WRAPPER ---
class VizAgent:
    def __init__(self, model_path, strategy="standard", device="cpu"):
        """
        strategy: 
          - "standard": Plays normally (used for Right side).
          - "mirror": Flips observation horizontally (used for Left side with Right model).
          - "generalist": Plays normally but expects the model to handle Left side logic naturally.
        """
        print(f"Loading model: {model_path} ({strategy})")
        self.model = PPO.load(model_path, device=device)
        self.strategy = strategy

    def get_action(self, obs):
        # 1. Pre-process based on strategy
        if self.strategy == "mirror":
            # Flip width (axis 2)
            obs = np.flip(obs, axis=2)
            # Correct the shift caused by flipping (alignment fix)
            obs = np.roll(obs, shift=-2, axis=2)
            # Clean artifacts on edge
            obs[:, :, -2:] = 0
        
        # 2. Predict
        action, _ = self.model.predict(obs, deterministic=True)
        return int(action)

# --- 3. GAME LOOP ---
def run_game(left_agent, right_agent, title="Game Test"):
    env = make_viz_env(render_mode="human")
    env.reset()
    
    print(f"\n--- STARTING: {title} ---")
    print("Left Paddle: Controlled by", left_agent.strategy)
    print("Right Paddle: Controlled by", right_agent.strategy)
    
    total_reward_left = 0
    total_reward_right = 0
    
    # Run one episode
    for agent in env.agent_iter():
        obs, reward, termination, truncation, info = env.last()
        
        # Track score
        if agent == "first_0": # Left Agent
            total_reward_left += reward
        else: # Right Agent
            total_reward_right += reward

        if termination or truncation:
            action = None
        else:
            if agent == "first_0": # Left Player
                action = left_agent.get_action(obs)
            else: # Right Player
                action = right_agent.get_action(obs)
        
        env.step(action)
        
        # Slow down slightly for visibility if needed
        # time.sleep(0.01)

    print(f"GAME OVER. Score - Left: {total_reward_left} | Right: {total_reward_right}")
    env.close()

# --- 4. MAIN EXECUTION ---
if __name__ == "__main__":
    
    # ---------------------------------------------------------
    # CONFIGURATION: Update these paths to your actual files
    # ---------------------------------------------------------
    
    # PATH 1: Your Best "Part 1" Model (Right-side specialist)
    # This is used for the "Right" side in all tests, and for the "Left" side in the Mirror test.
    PATH_RIGHT_MODEL = "./exports/pong/best_model/best_model.zip"
    
    # PATH 2: Your "Left Specialist" Model (from train_left.py)
    # Only used if you trained a dedicated left agent.
    PATH_LEFT_SPECIALIST = "./exports/pong_left/best_model/best_model.zip"
    
    # PATH 3: Your "Generalist" Model (from train_generalist.py)
    # Only used if you ran the self-play script.
    PATH_GENERALIST = "./exports/pong_generalist/best_model/best_model.zip"

    # ---------------------------------------------------------
    # UNCOMMENT THE TEST YOU WANT TO RUN
    # ---------------------------------------------------------

    # === TEST A: MIRROR STRATEGY (Recommended) ===
    # Uses the SAME Right-side model for both, but flips the Left agent's view.
    print("\n>>> TEST A: MIRROR STRATEGY")
    left_bot = VizAgent(PATH_RIGHT_MODEL, strategy="mirror")
    right_bot = VizAgent(PATH_RIGHT_MODEL, strategy="standard")
    run_game(left_bot, right_bot, title="Mirror Strategy (Same Model)")


    # === TEST B: SPECIALIST vs SPECIALIST ===
    # Uses a dedicated Left model vs a dedicated Right model.
    # print("\n>>> TEST B: SPECIALIST STRATEGY")
    # try:
    #     left_bot = VizAgent(PATH_LEFT_SPECIALIST, strategy="standard") # Standard because it was TRAINED on the left
    #     right_bot = VizAgent(PATH_RIGHT_MODEL, strategy="standard")
    #     run_game(left_bot, right_bot, title="Specialist vs Specialist")
    # except Exception as e:
    #     print(f"Skipping Test B: Could not load Left Specialist model. ({e})")


    # === TEST C: GENERALIST (SELF-PLAY) ===
    # Uses one model that learned to play both sides natively.
    # print("\n>>> TEST C: GENERALIST STRATEGY")
    # try:
    #     left_bot = VizAgent(PATH_GENERALIST, strategy="generalist")
    #     right_bot = VizAgent(PATH_GENERALIST, strategy="generalist")
    #     run_game(left_bot, right_bot, title="Generalist Self-Play")
    # except Exception as e:
    #     print(f"Skipping Test C: Could not load Generalist model. ({e})")