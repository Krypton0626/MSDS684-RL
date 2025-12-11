"""
Visualization module for Lab 8 – Modern Deep RL Exploration.

Generates:
1. Learning rate effects plot
2. Architecture + batch size ablation plot

Uses results saved in lab8/data/ppo_results.npy
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
FIG_DIR = ROOT_DIR / "figs"

DATA_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)


def plot_lr_effects():
    """
    Plot learning rate effects for PPO on CartPole-v1.
    Uses fixed: net_arch=(64,64), batch_size=64
    """

    results_path = DATA_DIR / "ppo_results.npy"
    results = np.load(results_path, allow_pickle=True)

    # Filter for: arch=(64,64) & batch_size=64
    filtered = [
        r for r in results
        if tuple(r["net_arch"]) == (64, 64) and r["batch_size"] == 64
    ]

    # Group by learning rate
    groups = {}
    for r in filtered:
        lr = r["learning_rate"]
        if lr not in groups:
            groups[lr] = []
        groups[lr].append(r)

    plt.figure(figsize=(10, 7))

    for lr, runs in groups.items():
        # Stack timesteps and rewards
        # All runs share the same timesteps
        timesteps = runs[0]["timesteps"]
        mean_rewards = np.stack([run["eval_mean_rewards"] for run in runs])
        std_rewards = np.stack([run["eval_std_rewards"] for run in runs])

        # Mean over seeds
        mean_curve = mean_rewards.mean(axis=0)
        std_curve = mean_rewards.std(axis=0)

        plt.plot(timesteps, mean_curve, label=f"lr={lr:.0e}")
        plt.fill_between(
            timesteps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
        )

    plt.title(
        "PPO on CartPole-v1: Learning Rate Effects\n(net_arch=64-64, batch_size=64)",
        fontsize=14,
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean evaluation reward")
    plt.legend()
    plt.grid(True)

    out_path = FIG_DIR / "lr_effects.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_arch_batch_ablation():
    """
    Plot net architecture vs batch size effects on PPO for CartPole-v1.
    Uses a fixed lr = 3e-4 for comparison.
    """

    results_path = DATA_DIR / "ppo_results.npy"
    results = np.load(results_path, allow_pickle=True)

    target_lr = 3e-4
    filtered = [
        r for r in results
        if r["learning_rate"] == target_lr
    ]

    # Group by (net_arch, batch_size)
    groups = {}
    for r in filtered:
        key = (tuple(r["net_arch"]), r["batch_size"])
        if key not in groups:
            groups[key] = []
        groups[key].append(r)

    plt.figure(figsize=(10, 7))

    for (arch, batch), runs in groups.items():
        timesteps = runs[0]["timesteps"]
        mean_rewards = np.stack([run["eval_mean_rewards"] for run in runs])
        std_rewards = np.stack([run["eval_std_rewards"] for run in runs])

        mean_curve = mean_rewards.mean(axis=0)
        std_curve = mean_rewards.std(axis=0)

        label = f"arch={arch}, batch={batch}"
        plt.plot(timesteps, mean_curve, label=label)
        plt.fill_between(
            timesteps,
            mean_curve - std_curve,
            mean_curve + std_curve,
            alpha=0.2,
        )

    plt.title(
        "PPO on CartPole-v1: Net Architecture & Batch Size (lr=3e-4)",
        fontsize=14,
    )
    plt.xlabel("Timesteps")
    plt.ylabel("Mean evaluation reward")
    plt.legend()
    plt.grid(True)

    out_path = FIG_DIR / "arch_batch_ablation.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


if __name__ == "__main__":
    plot_lr_effects()
    plot_arch_batch_ablation()
    print("All plots saved!")
