"""
run_lab3.py — Minimal runner for Lab 3 (Monte Carlo Control)

Runs the Blackjack-v1 Monte Carlo training script,
saves Q-values and episode return arrays, and prints status updates.
"""

import numpy as np
from lab3.mc_blackjack import mc_control_first_visit
import gymnasium as gym
import os

def main():
    # Create output directory
    os.makedirs("lab3/data", exist_ok=True)

    # Environment
    env = gym.make("Blackjack-v1")

    # Training parameters
    episodes = 500_000
    epsilon_start = 1.0
    epsilon_min = 0.02
    epsilon_decay = 0.99995
    gamma = 1.0

    print(f"Running Monte Carlo Control for {episodes:,} episodes...")
    Q, episode_returns = mc_control_first_visit(
        env=env,
        num_episodes=episodes,
        gamma=gamma,
        epsilon_start=epsilon_start,
        epsilon_min=epsilon_min,
        epsilon_decay=epsilon_decay
    )

    # Save outputs
    np.save("lab3/data/Q_blackjack.npy", Q)
    np.save("lab3/data/episode_returns.npy", episode_returns)

    print("Training complete!")
    print("Saved:")
    print(" - lab3/data/Q_blackjack.npy")
    print(" - lab3/data/episode_returns.npy")
    print("You can now run:  python viz_blackjack.py  to generate figures.")

if __name__ == "__main__":
    main()
