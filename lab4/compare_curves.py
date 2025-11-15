import numpy as np
from pathlib import Path

from .utils_td import plot_learning_curves

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIGS_DIR = BASE_DIR / "figs"


def main():
    sarsa_path = DATA_DIR / "episode_returns_sarsa.npy"
    q_path = DATA_DIR / "episode_returns_qlearning.npy"

    if not sarsa_path.exists() or not q_path.exists():
        raise FileNotFoundError(
            "Saved returns not found. Run cliff_sarsa_q.py first to generate data."
        )

    sarsa_returns = np.load(sarsa_path)
    q_returns = np.load(q_path)

    plot_learning_curves(
        sarsa_returns,
        q_returns,
        out_path=FIGS_DIR / "learning_curves_td_from_saved.png",
        title="CliffWalking: SARSA vs Q-learning (from saved arrays)",
    )

    print("Re-generated learning curve from saved data.")


if __name__ == "__main__":
    main()
