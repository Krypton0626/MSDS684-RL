# Lab 8 — Modern Deep Reinforcement Learning Exploration (PPO)

This lab examines contemporary deep reinforcement learning in application, utilizing the Proximal Policy Optimization (PPO) algorithm from Stable-Baselines3 (SB3).  
Instead of writing algorithms from scratch, the focus is on figuring out **why modern RL methods are stable**, how **engineering choices matter**, and how **hyperparameters affect learning behavior**.

Experiments are conducted on **CartPole-v1** and examine:
- Learning rate sensitivity
- Policy/value network size
- Batch-size ablation
- Performance variability across random seeds

This work corresponds to **Option C: Modern Deep RL Exploration**.

---

## How to Run

### 1. Create Virtual Environment


python -m venv .venv
.venv\Scripts\activate   # Windows

### 2. Install Dependencies

pip install -r requirements.txt

Note: LunarLander-v2 could not be used due to Box2D build issues on Windows with Python 3.13.
CartPole-v1 is a supported Option C environment and was selected instead.

### 3. Run PPO Experiments

3. Run PPO Experiments

This executes:

- A learning-rate sweep

- Two network architectures

- A batch-size ablation

- Multiple random seeds per configuration

## Outputs

### Data

- Saved to: lab8/data/

- Includes evaluation returns and aggregated statistics.

### Visualizations

- Saved to: lab8/figs/

### Key plots:

- Learning-rate comparison (mean ± std)

- Architecture and batch-size ablation

### Visualizations can be regenerated with:

- python lab8/viz.py

### Environment Summary (CartPole-v1)

| Component   | Description                    |
| ----------- | ------------------------------ |
| State Space | Continuous, 4-dimensional      |
| Actions     | Discrete (left, right)         |
| Reward      | +1 per time step balanced      |
| Termination | Angle, position, or time limit |
| Algorithm   | PPO (actor–critic)             |


## Summary

- PPO works well to solve CartPole-v1 in a lot of different situations.

- The learning rate has the biggest effect on how quickly and stably things come together.

- For simple tasks, larger networks don't always get better results.

- The size of the batch controls the balance between updates that are noisy and those that are stable.

- Not just algorithms, but also engineering choices are important for modern RL to work.