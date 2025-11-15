# Lab 4 — Temporal-Difference Learning (SARSA & Q-Learning)

This lab implements two TD control algorithms — SARSA (on-policy) and Q-learning (off-policy) — in the CliffWalking-v1 environment. The goal is to compare how each method learns under exploration and how on-policy vs off-policy updates affect the final learned behavior.

## How to Run

### 1. Set up environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

### 2. Run TD experiments
python run_lab4.py
Outputs saved to /lab4/data:
- episode_returns_sarsa.npy
- episode_returns_qlearning.npy
- Q_sarsa.npy
- Q_qlearning.npy

### 3. Generate visualizations
Saved to /lab4/figs:
- learning_curves_td.png
- value_sarsa.png
- value_qlearning.png
- policy_sarsa.png
- policy_qlearning.png
- trajectory_sarsa.png
- trajectory_qlearning.png

## Environment Summary
- Env: CliffWalking-v1  
- Reward: step = -1, falling into cliff = -100  
- Actions: 4 (up, right, down, left)  
- States: 48 grid positions  
- Episodes end on reaching the goal or falling from the cliff  
- Gamma: 1.0

## Summary
SARSA learned a safe path by accounting for exploration risk, while Q-learning learned the optimal but risky cliff-edge path. TD updates allowed both algorithms to learn online after every step, showing faster convergence than Monte Carlo methods and highlighting the difference between on-policy and off-policy learning.
