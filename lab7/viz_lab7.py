"""
viz_lab7.py

Visualization helpers for Lab 7:

- Q-learning vs Dyna-Q (different planning steps)
- Dyna-Q vs Dyna-Q+ in a dynamic environment
- Uniform Dyna-Q vs Prioritized Sweeping

Assumes that runner_lab7.py has already saved .npy result files into lab7/data/.
"""

from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from .utils_lab7 import DATA_DIR, FIGS_DIR, make_dirs


def _load_returns(filenames: List[str]) -> Dict[str, np.ndarray]:
    """
    Load episodic returns arrays from DATA_DIR.

    Each file is expected to have shape [n_seeds, n_episodes].

    Returns
    -------
    dict : name -> np.ndarray
        Where name is derived from filename (without extension).
    """
    results = {}
    for fname in filenames:
        path = DATA_DIR / fname
        arr = np.load(path)
        name = fname.replace(".npy", "")
        results[name] = arr
    return results


def _plot_with_ci(
    x: np.ndarray,
    series: Dict[str, np.ndarray],
    title: str,
    xlabel: str,
    ylabel: str,
    outfile: str,
):
    """
    Plot mean ± 95% CI for each series over x.
    """
    make_dirs()

    plt.figure()
    for label, values in series.items():
        # values: shape [n_seeds, n_steps]
        mean = values.mean(axis=0)
        std = values.std(axis=0)
        n_seeds = values.shape[0]
        ci = 1.96 * std / np.sqrt(max(n_seeds, 1))

        plt.plot(x, mean, label=label)
        plt.fill_between(x, mean - ci, mean + ci, alpha=0.2)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()

    out_path = FIGS_DIR / outfile
    plt.savefig(out_path)
    plt.close()
    print(f"Saved figure to {out_path}")


# ---------------------------------------------------------------------
# Public plotting functions
# ---------------------------------------------------------------------


def plot_q_vs_dyna():
    """
    Plot average episodic returns for:
      - Q-learning (n = 0)
      - Dyna-Q with n in {5, 10, 50}
    """
    fnames = [
        "q_learning_returns.npy",
        "dyna_q_n5_returns.npy",
        "dyna_q_n10_returns.npy",
        "dyna_q_n50_returns.npy",
    ]
    results = _load_returns(fnames)

    # Assume all have same number of episodes
    n_episodes = next(iter(results.values())).shape[1]
    x = np.arange(n_episodes)

    # Rename keys for prettier legend
    series = {
        "Q-learning (n=0)": results["q_learning_returns"],
        "Dyna-Q (n=5)": results["dyna_q_n5_returns"],
        "Dyna-Q (n=10)": results["dyna_q_n10_returns"],
        "Dyna-Q (n=50)": results["dyna_q_n50_returns"],
    }

    _plot_with_ci(
        x=x,
        series=series,
        title="Taxi-v3: Q-learning vs Dyna-Q (Episodic Returns)",
        xlabel="Episode",
        ylabel="Return",
        outfile="lab7_q_vs_dyna_ci.png",
    )


def plot_dyna_q_vs_dyna_q_plus_dynamic():
    """
    Plot returns for Dyna-Q vs Dyna-Q+ in the dynamic environment.
    """
    fnames = [
        "dyna_q_dynamic_returns.npy",
        "dyna_q_plus_dynamic_returns.npy",
    ]
    results = _load_returns(fnames)

    n_episodes = next(iter(results.values())).shape[1]
    x = np.arange(n_episodes)

    series = {
        "Dyna-Q (dynamic env)": results["dyna_q_dynamic_returns"],
        "Dyna-Q+ (dynamic env)": results["dyna_q_plus_dynamic_returns"],
    }

    _plot_with_ci(
        x=x,
        series=series,
        title="Dyna-Q vs Dyna-Q+ in Dynamic Taxi Environment",
        xlabel="Episode",
        ylabel="Return",
        outfile="lab7_dyna_q_vs_dyna_q_plus_dynamic_ci.png",
    )


def plot_uniform_vs_prioritized():
    """
    Plot returns for uniform Dyna-Q vs prioritized sweeping.
    """
    fnames = [
        "dyna_q_uniform_returns.npy",
        "prioritized_sweeping_returns.npy",
    ]
    results = _load_returns(fnames)

    n_episodes = next(iter(results.values())).shape[1]
    x = np.arange(n_episodes)

    series = {
        "Dyna-Q (uniform planning)": results["dyna_q_uniform_returns"],
        "Prioritized Sweeping": results["prioritized_sweeping_returns"],
    }

    _plot_with_ci(
        x=x,
        series=series,
        title="Uniform Dyna-Q vs Prioritized Sweeping",
        xlabel="Episode",
        ylabel="Return",
        outfile="lab7_uniform_vs_prioritized_ci.png",
    )


def plot_nn_model_based_pg():
    """
    Plot the returns of model-based policy gradient trained on the learned neural dynamics model.
    """
    path = DATA_DIR / "nn_model_based_pg_returns.npy"
    if not path.exists():
        print("No neural dynamics PG results found; skipping plot.")
        return

    returns = np.load(path)

    plt.figure(figsize=(8,5))
    plt.plot(returns, label="Model-Based PG (simulated)")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Neural Dynamics Model-Based Policy Gradient Returns")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    out_path = FIGS_DIR / "nn_model_based_pg.png"
    plt.savefig(out_path)
    plt.close()

    print(f"Saved: {out_path}")


def run_all_plots():
    """
    Convenience function: generate all Lab 7 plots after experiments
    have been run and .npy files exist in lab7/data/.
    """
    plot_q_vs_dyna()
    plot_dyna_q_vs_dyna_q_plus_dynamic()
    plot_uniform_vs_prioritized()
    plot_nn_model_based_pg()