import numpy as np
import gymnasium as gym
from pathlib import Path

from td_algos import sarsa, q_learning
from utils_td import (
    plot_learning_curves,
    plot_value_heatmap,
    plot_policy_arrows,
    plot_greedy_trajectory,
)

BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
FIGS_DIR = BASE_DIR / "figs"


def run_td_experiments(
    num_episodes=500,
    num_seeds=30,
    alpha=0.5,
    gamma=1.0,
    epsilon=0.1,
):
    """
    Run multi-seed SARSA and Q-learning on CliffWalking-v0.
    Save episode return arrays and generate plots.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)

    env_name = "CliffWalking-v0"

    sarsa_returns_all = np.zeros((num_seeds, num_episodes), dtype=np.float64)
    qlearning_returns_all = np.zeros((num_seeds, num_episodes), dtype=np.float64)

    Q_sarsa_last = None
    Q_q_last = None

    for seed in range(num_seeds):
        print(f"[Seed {seed}] Running SARSA...")
        env = gym.make(env_name)
        # Seed env once per run (not every episode)
        env.reset(seed=seed)

        Q_sarsa, returns_sarsa = sarsa(
            env,
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            seed=seed,
        )
        env.close()

        sarsa_returns_all[seed] = returns_sarsa
        Q_sarsa_last = Q_sarsa

        print(f"[Seed {seed}] Running Q-learning...")
        env = gym.make(env_name)
        env.reset(seed=seed)

        Q_q, returns_q = q_learning(
            env,
            num_episodes=num_episodes,
            alpha=alpha,
            gamma=gamma,
            epsilon=epsilon,
            seed=seed,
        )
        env.close()

        qlearning_returns_all[seed] = returns_q
        Q_q_last = Q_q

    # Save episode returns
    np.save(DATA_DIR / "episode_returns_sarsa.npy", sarsa_returns_all)
    np.save(DATA_DIR / "episode_returns_qlearning.npy", qlearning_returns_all)

    # Learning curves with 95% CI
    plot_learning_curves(
        sarsa_returns_all,
        qlearning_returns_all,
        out_path=FIGS_DIR / "learning_curves_td.png",
        title="CliffWalking: SARSA vs Q-learning (mean ± 95% CI)",
    )

    # Value function heatmaps
    shape = (4, 12)
    plot_value_heatmap(
        Q_sarsa_last,
        shape=shape,
        out_path=FIGS_DIR / "value_sarsa.png",
        title="V(s) from SARSA (greedy)",
    )
    plot_value_heatmap(
        Q_q_last,
        shape=shape,
        out_path=FIGS_DIR / "value_qlearning.png",
        title="V(s) from Q-learning (greedy)",
    )

    # Greedy policies
    plot_policy_arrows(
        Q_sarsa_last,
        shape=shape,
        out_path=FIGS_DIR / "policy_sarsa.png",
        title="Greedy policy from SARSA",
    )
    plot_policy_arrows(
        Q_q_last,
        shape=shape,
        out_path=FIGS_DIR / "policy_qlearning.png",
        title="Greedy policy from Q-learning",
    )

    # Sample trajectories
    plot_greedy_trajectory(
        env_name,
        Q_sarsa_last,
        max_steps=100,
        out_path=FIGS_DIR / "trajectory_sarsa.png",
        title="Sample greedy trajectory (SARSA Q)",
    )
    plot_greedy_trajectory(
        env_name,
        Q_q_last,
        max_steps=100,
        out_path=FIGS_DIR / "trajectory_qlearning.png",
        title="Sample greedy trajectory (Q-learning Q)",
    )

    print("TD experiment complete.")
    print(f"Saved returns to: {DATA_DIR}")
    print(f"Saved figures to: {FIGS_DIR}")


def main():
    # For a first test, you can temporarily lower these:
    # run_td_experiments(num_episodes=100, num_seeds=5)
    run_td_experiments(num_episodes=500, num_seeds=30)


if __name__ == "__main__":
    main()
