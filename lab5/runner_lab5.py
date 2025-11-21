# lab5/runner_lab5.py

import os
import numpy as np
import gymnasium as gym

from .mc_sarsa import semi_gradient_sarsa
from .viz_mountaincar import (
    plot_learning_curves,
    plot_value_function,
    plot_policy,
    plot_trajectories,
)


def main():
    """
    Train semi-gradient SARSA agents with different tile-coding configs
    on MountainCar-v0, save learning curves and visualizations.
    """
    os.makedirs("lab5/data", exist_ok=True)
    os.makedirs("lab5/figs", exist_ok=True)

    # Config label -> (num_tilings, tiles_per_dim)
    configs = [
        ("8tilings_8x8", dict(num_tilings=8, tiles_per_dim=(8, 8))),
        ("4tilings_4x4", dict(num_tilings=4, tiles_per_dim=(4, 4))),
        ("8tilings_16x16", dict(num_tilings=8, tiles_per_dim=(16, 16))),
    ]

    learning_curves = []

    for label, cfg in configs:
        print(f"\n=== Training config: {label} ===")
        w, lengths, tile_coder = semi_gradient_sarsa(
            env_name="MountainCar-v0",
            num_episodes=3000,
            alpha=0.1,          # internally scaled by num_tilings
            gamma=1.0,
            epsilon_start=1.0,
            epsilon_end=0.05,
            **cfg,
        )

        # Save episode lengths
        np.save(f"lab5/data/lengths_{label}.npy", np.array(lengths))

        # For the “main” config, generate detailed visualizations
        if label == "8tilings_8x8":
            plot_value_function(
                w,
                tile_coder,
                save_path="lab5/figs/value_heatmap_8tilings_8x8.png",
            )
            plot_policy(
                w,
                tile_coder,
                save_path="lab5/figs/policy_8tilings_8x8.png",
            )
            env = gym.make("MountainCar-v0")
            plot_trajectories(
                w,
                tile_coder,
                env,
                n_episodes=5,
                save_path="lab5/figs/trajectories_8tilings_8x8.png",
            )
            env.close()

        learning_curves.append((label, lengths))

    # Combined learning curves
    plot_learning_curves(
        learning_curves,
        save_path="lab5/figs/learning_curves_all.png",
    )

    print("\n[Lab 5] Done. Check lab5/data/ and lab5/figs/ for outputs.")
