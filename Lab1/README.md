Lab 1 — Multi-Armed Bandits and MDP Foundations

Goals:
- Implement a custom 10-armed bandit environment with a Gymnasium-like API.
- Implement ε-greedy and UCB agents in NumPy.
- Run 1000 runs × 2000 steps and record:
  - average reward over time
  - % optimal action over time
- Explore Gymnasium environments (FrozenLake-v1, Taxi-v3) and relate those to the MDP tuple (S, A, P, R, γ).

Reproduction steps :
1. Activate the virtual environment (venv).
2. pip install -r requirements.txt
3. Run `python bandit_experiments.py` → generates plots into `lab1/figs/`
4. Run `python gym_exploration.py` → inspects FrozenLake-v1 and Taxi-v3 and prints MDP structure
