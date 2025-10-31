# Lab 2 — Dynamic Programming and Value Function Estimation

## Goals
- Implement a configurable **GridWorld** environment with deterministic and stochastic transitions.
- Implement and compare **Policy Evaluation (synchronous & in-place)**, **Policy Iteration**, and **Value Iteration** algorithms using NumPy.
- Visualize:
  - State-value heatmaps
  - Greedy-policy arrows
  - Convergence curves (Δ vs. sweeps)
- Reproduce results on **FrozenLake-v1** (slippery and non-slippery) using `env.unwrapped.P`.

## Reproduction Steps
1. Activate your virtual environment.

   - conda create -n rl_lab2 python=3.10 -y
   - conda activate rl_lab2
   - pip install -r requirements.txt

2. Run the experiments

    - python experiments/run_gridworld.py
    - python experiments/run_frozenlake.py

3. Generated Figures will appear in the /results directory:

    - gridworld_values.png
    - gridworld_policy.png
    - delta_convergence.png
    - frozenlake_values.png

