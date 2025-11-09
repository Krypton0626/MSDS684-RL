# Lab 3 — Monte Carlo Methods and On-Policy Control  

## Goals  
- Implement **first-visit Monte Carlo control** on Gymnasium’s `Blackjack-v1`.  
- Explore **ε-soft policies** with **ε-decay** for exploration.  
- Estimate **Q(s,a)** using **incremental averaging** (model-free).  
- Visualize:  
  - 3D **value surfaces** (usable vs. non-usable ace)  
  - **Policy heatmaps** (hit/stick boundaries)  
  - **Learning curve** of average returns.  
- Compare learned strategies with **basic Blackjack strategy** and discuss convergence.  

---

## Reproduction Steps  

1. **Setup Environment**  
   ```bash
   python -m venv .venv
   source .venv/bin/activate       
   pip install -r requirements.txt

2. **Train the Agent**
    python mc_blackjack.py

- Runs 500 000 episodes (ε = 1.0 → 0.02).
- Outputs saved in /data:
    Q_blackjack.npy
    episode_returns.npy

3. **Generate Visuals**

    python viz_blackjack.py

**Figures in /figs:**

- value_surface_usable_ace.png
- value_surface_no_ace.png
- policy_usable_ace.png
- policy_no_ace.png
- learning_curve.png

** Environment**

| Component | Description                                    |
| :-------- | :--------------------------------------------- |
| Env       | `Blackjack-v1`                                 |
| Reward    | +1 win, 0 draw, −1 loss                        |
| State     | (player_sum 4–21, dealer 1–10, usable_ace T/F) |
| Actions   | 0 = stick, 1 = hit                             |
| γ         | 1.0                                            |
