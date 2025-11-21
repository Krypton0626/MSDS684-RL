# Lab 5 — Function Approximation with Tile Coding (Semi-Gradient SARSA)

This lab extends reinforcement learning from tabular methods to **function approximation**, using **tile coding** and **semi-gradient SARSA** to solve the continuous-state MountainCar-v0 environment. The goal is to show how feature-based value approximation enables RL to scale beyond finite state spaces.

---

## How to Run

### **1. Set up environment**


python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r lab5/requirements.txt

### **2. Run Experiments**

python run_lab5.py

Outputs saved to /lab5/data:

    - lengths_8tilings_8x8.npy

    - lengths_4tilings_4x4.npy

    - lengths_8tilings_16x16.npy

* Weight vectors for each configuration

### **3. Generate visualizations**

Saved to /lab5/figs:

    - learning_curves_all.png

    - value_heatmap_8tilings_8x8.png

    - policy_8tilings_8x8.png

    - trajectories_8tilings_8x8.png

### **Environment Summary**

    * Env: MountainCar-v0
    * State: Continuous [𝑝𝑜𝑠𝑖𝑡𝑖𝑜𝑛,𝑣𝑒𝑙𝑜𝑐𝑖𝑡𝑦]
    * Actions: 3 (push left, no push, push right)
    * nReward: –1 each step
    * Goal: Reach position ≥ 0.5
    * Episode Limit: 200 steps
    * Gamma: 1.0

### **Tile Coding Configurations Tested:**

    - 8 tilings × 8×8 tiles

    - 4 tilings × 4×4 tiles

    - 8 tilings × 16×16 tiles

### **Summary**

Semi-gradient SARSA with tile-coded features successfully learned the momentum-building strategy required to solve MountainCar.

### **Key findings:**

- 8 tilings × 8×8 tiles learned fastest and reached the best performance (~115 steps)

- 4×4 tiles generalized too broadly and underfit the environment.

- 16×16 tiles were more expressive but required more episodes to converge.

- Tile coding enabled local generalization, smooth value estimates, and stable learning — demonstrating why function approximation is essential for continuous RL tasks.