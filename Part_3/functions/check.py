import matplotlib.pyplot as plt
from preprocessing_aina import make_env

env = make_env("ALE/Skiing-v5")
obs, _ = env.reset()

# Unstack the frames to see the 4th frame (most recent)
# obs shape is likely (4, 84, 84)
plt.imshow(obs[3], cmap='gray') 
plt.show()