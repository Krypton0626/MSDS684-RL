# lab6/runner_lab6.py

import os

import matplotlib.pyplot as plt
import numpy as np

# Support both running as a package and as a script
try:
    from .reinforce_cartpole import run_multi_seed
    from .utils import ensure_dir, save_returns, plot_with_ci
except ImportError:
    from reinforce_cartpole import run_multi_seed
    from utils import ensure_dir, save_returns, plot_with_ci


def main():
    # Base dir = this file's folder (lab6/)
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, "data")
    figs_dir = os.path.join(base_dir, "figs")

    ensure_dir(data_dir)
    ensure_dir(figs_dir)

    num_seeds = 10
    num_episodes = 500
    gamma = 0.99

    # 1. REINFORCE (no baseline)
    print("Running REINFORCE (no baseline)...")
    returns_reinforce, entropy_reinforce = run_multi_seed(
        algo="reinforce",
        num_seeds=num_seeds,
        num_episodes=num_episodes,
        gamma=gamma,
    )
    save_returns(
        os.path.join(data_dir, "returns_reinforce.npy"),
        np.asarray(returns_reinforce),
    )
    save_returns(
        os.path.join(data_dir, "entropy_reinforce.npy"),
        np.asarray(entropy_reinforce),
    )

    # 2. REINFORCE with baseline
    print("Running REINFORCE with baseline...")
    returns_baseline, entropy_baseline = run_multi_seed(
        algo="reinforce_baseline",
        num_seeds=num_seeds,
        num_episodes=num_episodes,
        gamma=gamma,
    )
    save_returns(
        os.path.join(data_dir, "returns_reinforce_baseline.npy"),
        np.asarray(returns_baseline),
    )
    save_returns(
        os.path.join(data_dir, "entropy_reinforce_baseline.npy"),
        np.asarray(entropy_baseline),
    )

    # 3. Learning curve with 95% CI (already saw this)
    plt.figure(figsize=(8, 5))
    plot_with_ci(returns_reinforce, label="REINFORCE", color="tab:blue")
    plot_with_ci(returns_baseline, label="REINFORCE + baseline", color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Return (per episode)")
    plt.title("CartPole-v1: REINFORCE vs REINFORCE + baseline")
    plt.legend()
    plt.tight_layout()
    fig1_path = os.path.join(figs_dir, "cartpole_reinforce_baseline_ci.png")
    plt.savefig(fig1_path, dpi=200)
    plt.close()

    # 4. Policy entropy curves (mean ± CI)
    plt.figure(figsize=(8, 5))
    plot_with_ci(entropy_reinforce, label="REINFORCE", color="tab:blue")
    plot_with_ci(entropy_baseline, label="REINFORCE + baseline", color="tab:orange")
    plt.xlabel("Episode")
    plt.ylabel("Policy entropy")
    plt.title("CartPole-v1: Policy Entropy vs Episodes")
    plt.legend()
    plt.tight_layout()
    fig2_path = os.path.join(figs_dir, "policy_entropy_ci.png")
    plt.savefig(fig2_path, dpi=200)
    plt.close()

    print("Done. Saved:")
    print(f"  {os.path.join(data_dir, 'returns_reinforce.npy')}")
    print(f"  {os.path.join(data_dir, 'returns_reinforce_baseline.npy')}")
    print(f"  {os.path.join(data_dir, 'entropy_reinforce.npy')}")
    print(f"  {os.path.join(data_dir, 'entropy_reinforce_baseline.npy')}")
    print(f"  {fig1_path}")
    print(f"  {fig2_path}")


if __name__ == "__main__":
    main()
