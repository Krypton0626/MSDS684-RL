"""
Root entry point for Lab 8 – Modern Deep RL Exploration.

Usage:
    python run_lab8.py

This will:
1. Run PPO hyperparameter experiments on CartPole-v1 using lab8/final_runner.py
2. Generate plots using lab8/viz.py
"""

from lab8.final_runner import main as run_experiments
from lab8.viz import plot_lr_effects, plot_arch_batch_ablation


def visualize_results():
    print("\n--- Generating visualizations for Lab 8 ---")
    plot_lr_effects()
    plot_arch_batch_ablation()
    print("Visualization complete. Figures saved under lab8/figs/.\n")


if __name__ == "__main__":
    # 1. Run PPO experiments and save results
    run_experiments()

    # 2. Generate learning curves and ablation plots
    visualize_results()
