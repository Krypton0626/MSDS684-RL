#  Lab 6 — Policy Gradient Methods (REINFORCE & Baseline)

This lab implements policy gradient reinforcement learning for the **CartPole-v1** environment using PyTorch.  
We compare:

- **Vanilla REINFORCE** — high-variance Monte Carlo policy gradient  
- **REINFORCE + Value Baseline** — variance-reduced and more stable  

The goal is to demonstrate how subtracting a learned baseline stabilizes learning and improves sample efficiency.

---

##  How to Run

### 1️. Create Virtual Environment

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r lab6/requirements.txt

### 2️. Run Experiments
python run_lab6.py


## Outputs saved to /lab6/data:

- returns_reinforce.npy

- returns_reinforce_baseline.npy

- entropy_reinforce.npy

- entropy_reinforce_baseline.npy

## Multiple random seeds are evaluated for stability.

### 3️. Generate Visualizations

## Plots saved to /lab6/figs:

- cartpole_reinforce_baseline_ci.png

- policy_entropy_ci.png

## These include confidence-interval learning curves and entropy decay for both algorithms.


### Environment Summary 

| Component               | Description                                                                |
| ----------------------- | -------------------------------------------------------------------------- |
| **Environment**         | CartPole-v1                                                                |
| **State Space**         | Continuous 4-D vector *(position, velocity, pole angle, angular velocity)* |
| **Actions**             | 2 *(left, right)*                                                          |
| **Reward**              | +1 per step                                                                |
| **Termination**         | Pole falls, cart out of bounds, or 500 steps                               |
| **Discount Factor (γ)** | 0.99                                                                       |

### Algorithms Implemented

## Vanilla REINFORCE

Monte Carlo policy gradient

Update rule:

−log𝜋(𝑎∣𝑠)𝐺𝑡​

- High variance and unstable gradients

- Sensitive to early-return noise

## REINFORCE + Value Baseline

Subtracts learned value estimate 𝑉(𝑠)

Uses advantage: 𝐺𝑡−𝑉(𝑠)

- Dramatically reduces variance

- Faster, smoother, and more consistent learning

### Summary

The baseline-enhanced REINFORCE algorithm learned CartPole much faster and with much less variation across seeds. It kept exploration healthier at first, which was clear from the smoother entropy decay, and it got close to the best returns in 100–150 episodes.

In contrast, vanilla REINFORCE was noisy, slower, and more unstable due to its high Monte Carlo variance. These findings corroborate Sutton and Barto's theoretical assertions concerning the significance of variance reduction in policy gradient methodologies.