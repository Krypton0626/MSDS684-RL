# Lab 7 — Deep Q-Learning - Planning and Learning Integration (Dyna-Q, Dyna-Q+, Prioritized Sweeping)

This lab investigates how planning and learning can be integrated using the Dyna architecture in the Taxi-v3 environment. The work compares:

- Q-Learning (no planning)
- Dyna-Q with different planning steps (n = 5, 10, 50)
- Dyna-Q+ with exploration bonuses in a dynamic environment
- Prioritized Sweeping for more efficient planning
- Optional: Neural Dynamics Model + Model-Based Policy Gradient (MountainCar-v0)

The goal is to demonstrate how combining model-free updates with simulated experience from a learned model improves sample efficiency, adaptation, and learning speed.

## How to Run

### 1. Create Virtual Environment

python -m venv .venv
..venv\Scripts\activate # Windows
cd lab7
pip install -r requirements.txt
cd ..


### 2. Run Experiments

python run_lab7.py


## Outputs are automatically saved to:

- lab7/data/

### These include episodic returns, episode lengths, and model-based RL results.

### 3. Generate Visualizations

Plots are saved to:

- lab7/figs/


These include confidence-interval learning curves comparing all algorithms:
- Q-learning vs Dyna-Q
- Dyna-Q vs Dyna-Q+ in dynamic Taxi
- Uniform Dyna-Q vs Prioritized Sweeping
- Neural model-based policy gradient (optional)

## Environment Summary (Taxi-v3)

| Component | Description |
|----------|-------------|
| Environment | Taxi-v3 (discrete, deterministic) |
| State Space | 500 discrete states |
| Actions | 6 (N, S, E, W, pickup, dropoff) |
| Reward | -1 per step, +20 on successful dropoff, -10 illegal move |
| Termination | Passenger delivered or step limit |
| Discount Factor (γ) | 0.99 |

Taxi-v3 is ideal for exploring planning because its dynamics can be stored perfectly in a dictionary-based tabular model.

## Algorithms Implemented

### Q-Learning (Baseline)
Pure model-free learning using temporal-difference updates.
No planning (n = 0).

### Dyna-Q
Model-free Q-learning combined with n simulated updates from a learned tabular model.
Planning budgets tested:  
`n ∈ {5, 10, 50}`

Expected effect: faster learning because each real experience generates multiple simulated updates.

### Dyna-Q+ (Dynamic Environment)
Extends Dyna-Q by adding exploration bonuses:

bonus = kappa * sqrt(tau)

where `tau` is the time since a state-action pair was last visited.

This helps adapt when the environment changes unexpectedly.

### Prioritized Sweeping
A more efficient planning method that:

- Uses a priority queue
- Updates transitions with the highest TD error first
- Propagates reward information faster than uniform replay

### Optional: Neural Dynamics + Model-Based Policy Gradient
For MountainCar-v0:
- Collect real transitions into a replay buffer
- Train a neural network to predict next state and reward
- Use simulated rollouts for REINFORCE policy updates

Produces:

- nn_model_based_pg_returns.npy
- nn_model_based_pg.png


## Summary

- When planning, Dyna-Q learns a lot faster than Q-learning by replaying stored transitions.  The most rapid convergence happens when the planning budgets are larger (n = 50).

- Dyna-Q+ adjusts to changes in the structure of the environment faster, showing how useful exploration bonuses can be in settings that aren't always the same.


- Prioritized Sweeping always does better than uniform planning because it puts updates where they will have the most effect.

- The optional neural dynamics experiment demonstrates that an imperfect learned model can facilitate policy enhancement via simulated rollouts, underscoring the applicability of Dyna concepts to continuous domains.
