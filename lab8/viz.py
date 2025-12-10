"""
Plotting utilities for Lab 8 PPO experiments.

Generates:
- lr_effects.png            : effect of learning rate (fixed net_arch, batch_size)
- arch_batch_ablation.png   : effect of network size and batch size (fixed lr)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import List, Dict, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .utils import group_by_keys

ROOT_DIR = Path(__file__).resolve().parent
DATA_PATH = ROOT_DIR / "data" / "ppo_results.npy"
FIG_DIR = ROOT_DIR / "figs"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def _load_results() -> List[Dict]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Results file not found at {DATA_PATH}. "
            "Run final_runner.py (or python run_lab8.py from project root) first."
        )
    arr = np.load(DATA_PATH, allow_pickle=True)
    return list(arr)


def plot_lr_effects():
    """
    Fix net_arch and batch_size, compare learning rates.
    """
    results = _load_results()

    target_arch = (64, 64)
    target_batch = 64

    filtered = [
        r for r in results
        if tuple(r["net_arch"]) == target_arch and int(r["batch_size"]) == target_batch
    ]

    if not filtered:
        print("No results found for lr plot; check that experiments ran with arch=(64,64), batch=64.")
        return

    grouped = group_by_keys(filtered, ("learning_rate",))

    plt.figure()
    for (lr,), runs in sorted(grouped.items(), key=lambda kv: kv[0][0]):
        timesteps = runs[0]["timesteps"]
        rewards = np.stack([run["eval_mean_rewards"] for run in runs], axis=0)

        mean_rewards = rewards.mean(axis=0)
        std_rewards = rewards.std(axis=0)

        plt.plot(timesteps, mean_rewards, label=f"lr={lr:.0e}")
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean evaluation reward")
    plt.title("PPO on LunarLander-v2: Learning Rate Effects\n(net_arch=64-64, batch_size=64)")
    plt.legend()
    out_path = FIG_DIR / "lr_effects.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def plot_arch_batch_ablation():
    """
    Fix learning rate, compare combinations of net_arch and batch_size.
    """
    results = _load_results()

    target_lr = 3e-4  # choose the "middle" LR for ablation
    filtered = [
        r for r in results
        if abs(float(r["learning_rate"]) - target_lr) < 1e-10
    ]

    if not filtered:
        print("No results found for arch/batch plot; check that experiments ran with lr=3e-4.")
        return

    grouped = group_by_keys(filtered, ("net_arch", "batch_size"))

    plt.figure()
    for (arch, batch), runs in grouped.items():
        arch_tuple = tuple(arch) if not isinstance(arch, tuple) else arch
        timesteps = runs[0]["timesteps"]
        rewards = np.stack([run["eval_mean_rewards"] for run in runs], axis=0)

        mean_rewards = rewards.mean(axis=0)
        std_rewards = rewards.std(axis=0)

        label = f"arch={arch_tuple}, batch={batch}"
        plt.plot(timesteps, mean_rewards, label=label)
        plt.fill_between(
            timesteps,
            mean_rewards - std_rewards,
            mean_rewards + std_rewards,
            alpha=0.2,
        )

    plt.xlabel("Timesteps")
    plt.ylabel("Mean evaluation reward")
    plt.title("PPO on LunarLander-v2: Net Architecture & Batch Size (lr=3e-4)")
    plt.legend()
    out_path = FIG_DIR / "arch_batch_ablation.png"
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved {out_path}")


def main():
    plot_lr_effects()
    plot_arch_batch_ablation()


if __name__ == "__main__":
    main()
