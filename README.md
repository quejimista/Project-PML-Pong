# Project Paradigms of Machine Learning

**Group 04** | Reinforcement Learning & Computer Vision

This repository contains the implementation and analysis of three distinct Reinforcement Learning (RL) paradigms. The project progresses from optimizing single-agent value-based methods to exploring multi-agent policy gradients, and finally addressing reward engineering in sparse reward environments.

## 📚 Project Structure

| Part | Domain | Environment | Key Algorithms |
| :--- | :--- | :--- | :--- |
| **Part 1** | Single Agent | `PongNoFrameskip-v4` | DQN, Double DQN, PER |
| **Part 2** | Multi-Agent | Competitive Pong (PettingZoo) | PPO, Self-Play, Bayesian Opt. |
| **Part 3** | Sparse Rewards | `ALE/Skiing-v5` | PPO, Reward Shaping (RAM) |

---

## Part 1: Single Agent Optimization (Pong)

**Objective:** Solve the Atari Pong environment using Deep Q-Networks (DQN) and optimize convergence and stability through advanced extensions.

### 🛠 Preprocessing Pipeline
To optimize the state space for the CNN, we implemented a strict preprocessing pipeline:
* **MaxAndSkip:** Skips 4 frames and selects the max value to capture dynamics.
* **Resize & Grayscale:** Downsamples input to `(84, 84, 1)`.
* **Frame Stacking:** Stacks the last 4 frames to provide temporal context `(4, 84, 84)`.
* **Normalization:** Scales pixel values to `[0, 1]`.

### 🧠 Models & Experiments
We implemented and compared four model configurations:
1.  **Standard DQN** (Baseline)
2.  **DQN + Prioritized Experience Replay (PER)**
3.  **Double DQN**
4.  **Double DQN + PER** (Selected Model)

**Key Insights:**
* **Double DQN:** Mitigated Q-value overestimation, leading to more stable learning.
* **PER:** Significantly improved data efficiency by prioritizing transitions with high Temporal Difference (TD) error.

### 🏆 Results
After an exhaustive hyperparameter search (tuning Learning Rate, Batch Size, and Epsilon Decay), the **Double DQN + PER** model achieved the best performance.

* **Best Configuration:** `Batch size: 32` | `Epsilon decay: 0.99`
* **Score:** Achieved a **perfect score of 21.0** (maximum possible reward).

---

## Part 2: Multi-Agent Competitive RL

**Objective:** Transition to a competitive multi-agent environment using **PettingZoo** and train a "Left Paddle Agent" to win a tournament using **Proximal Policy Optimization (PPO)**.

### ⚙️ Methodology
* **Preprocessing:** Custom wrappers were built to replicate `SuperSuit` behavior (Blue channel extraction, 84x84 resizing, Float32 conversion).
* **Hyperparameter Tuning:** Used **Bayesian Optimization** to efficiently tune PPO parameters (Learning Rate, Gamma, Entropy Coefficient).

### ⚔️ Tournament Strategies Explored
We attempted three distinct paradigms to create a competitive agent:

1.  **Mirroring:** Flipping observations from a trained Right Agent.
    * *Outcome:* **Failed.** Due to pixel offsets in the Pong rendering, the agent suffered from a "blind spot" and consistently missed returns.
2.  **Generalist Agent (Self-Play):** Training a single PPO model to play both sides simultaneously.
    * *Outcome:* **Failed.** The agent converged to a passive policy (zero-sum trap), learning to "exist" rather than win.
3.  **Fixed Opponent:** Training a Left Agent against a frozen, pre-trained Right Agent.
    * *Outcome:* **Unstable.** Encountered issues with reward logging and non-terminating states.

**Conclusion:** While our single-agent PPO achieved a 100% win rate against the built-in AI (Mean Reward: 18.05), the self-play strategies highlighted the difficulty of creating an auto-curriculum without an external expert reference.

---

## Part 3: Sparse Rewards (Atari Skiing)

**Objective:** Solve `ALE/Skiing-v5`, an environment characterized by extremely sparse rewards and long time horizons.

### 🧩 The Challenge
In *Skiing*, the agent only receives a negative reward for time penalties and a final reward at the end. Standard RL algorithms fail to learn because the causal link between actions (turning) and consequences (passing a flag) is too distant.

### 🔧 Solution: Reward Shaping via RAM
We engineered a dense reward function by directly accessing the Atari RAM states:
* **Orientation (RAM[15]):** Penalized horizontal movement to prevent the "suicide strategy" (crashing to end the episode early).
* **X-Position (RAM[25]):** Penalized staying near the edges.
* **Flags Remaining (RAM[107]):** Provided immediate positive reinforcement for passing flags.

### 📉 Hyperparameters & Results
* **Gamma:** Reduced to `0.9` to prioritize immediate shaped rewards.
* **Buffer Size:** Increased to `2048` to capture long trajectories.
* **Entropy Coefficient:** Tuned to `0.05` to balance exploration.

**Outcome:** The agent successfully learned to navigate the slope, either by optimizing for speed (straight line) or survival, effectively minimizing the environment's penalties.

---

## 🎥 Demos & Visualizations

Videos of the trained agents can be found in the respective folders:
* [Part 1 Videos](./Part_1/test_video)
* [Part 2 Videos](./Part_2/test_video)
* [Part 3 Videos](./Part_3/test_video)

## 💻 Installation

To reproduce the environments and training loops, install the required dependencies:

```bash
git clone https://github.com/quejimista/Project-PML-Pong.git
cd Project-PML-Pong
pip install gymnasium[atari] "gymnasium[accept-rom-license]" pettingzoo stable-baselines3 torch opencv-python
