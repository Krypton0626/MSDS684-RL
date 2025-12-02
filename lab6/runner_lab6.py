# lab6/runner_lab6.py

import os

import matplotlib.pyplot as plt
import numpy as np

from .reinforce_cartpole import run_multi_seed
from .utils import ensure_dir, save_returns, plot_with_ci


def main():
    # Base directory is this file's folder (lab6/)
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    figs_dir = os.path.join(base_dir, "figs")

    ensure_dir(data_dir)
    ensure_dir(figs_dir)

    num_seeds = 10
    num_episodes = 500
    gamma = 0.99

    print("Running REINFORCE (no baseline)...")
    returns_reinforce = run_multi_seed(
        algo="reinforce",
        num_seeds=num_seeds,
        num_episodes=num_episodes,
        gamma=gamma,
    )
    save_returns(os.path.join(data_dir, "returns_reinforce.npy"),
                 np.asarray(returns_reinforce))

    print("Running REINFORCE with baseline...")
    returns_baseline = run_multi_seed(
        algo="reinforce_baseline",
        num_seeds=num_seeds,
        num_episodes=num_episodes,
        gamma=gamma,
    )
    save_returns(os.path.join(data_dir, "returns_reinforce_baseline.npy"),
                 np.asarray(returns_baseline))

    # Plot learning curves with 95% CI
    plt.figure(figsize=(8, 5))
    plot_with_ci(returns_reinforce, label="REINFORCE", color="tab:blue")
    plot_with_ci(returns_baseline, label="REINFORCE + baseline", color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Return (per episode)")
    plt.title("CartPole-v1: REINFORCE vs REINFORCE + baseline")
    plt.legend()
    plt.tight_layout()

    fig_path = os.path.join(figs_dir, "cartpole_reinforce_baseline_ci.png")
    plt.savefig(fig_path, dpi=200)
    plt.close()

    print("Done. Saved:")
    print(f"  {os.path.join(data_dir, 'returns_reinforce.npy')}")
    print(f"  {os.path.join(data_dir, 'returns_reinforce_baseline.npy')}")
    print(f"  {fig_path}")


if __name__ == "__main__":
    main()
