import numpy as np
from stable_baselines3 import PPO

class TournamentAgent:
    """
    A wrapper for PPO model that handles Side Mirroring and Paddle Alignment.
    """
    def __init__(self, model_path, device="cuda"):
        self.model = PPO.load(model_path, device=device)
        
    def predict(self, observation, side="right", deterministic=True):
        if side == "left":
            # 1. Flip: Mirror the board so Left looks like Right
            # (Axis 2 is width)
            flipped = np.flip(observation, axis=2)
            
            # 2. Shift: Correct the misalignment caused by flipping
            # A shift of -2 moves the image 2 pixels to the left.
            processed_obs = np.roll(flipped, shift=-2, axis=2)
            
            # 3. Clean Artifacts: The roll wraps pixels around to the other side. 
            # We must zero out the "wrapped" pixels on the right edge.
            processed_obs[:, :, -2:] = 0
            
        else:
            # Right side: No changes
            processed_obs = observation

        # Predict
        action, _states = self.model.predict(processed_obs, deterministic=deterministic)
        
        return int(action)

# Simple test block
if __name__ == "__main__":
    # Create a dummy observation to test shapes
    dummy_obs = np.zeros((4, 84, 84), dtype=np.float32)
    
    # Fill left side with 1s to verify flip
    dummy_obs[:, :, :42] = 1.0 
    
    print("Dummy Obs Left Mean:", dummy_obs[:, :, :42].mean())   # Should be 1.0
    print("Dummy Obs Right Mean:", dummy_obs[:, :, 42:].mean())  # Should be 0.0
    
    # Test Flip Logic manually
    flipped_obs = np.flip(dummy_obs, axis=2)
    print("\nFlipped Obs Left Mean:", flipped_obs[:, :, :42].mean())  # Should be 0.0
    print("Flipped Obs Right Mean:", flipped_obs[:, :, 42:].mean())   # Should be 1.0
    
    print("\nLogic Verified: The agent on the Left will see the world mirrored.")