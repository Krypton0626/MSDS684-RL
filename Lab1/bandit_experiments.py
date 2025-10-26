import os
import sys

# --- PATH FIX ---
# This ensures the parent directory (the repo root) is on sys.path
# so that `import lab1.something` works no matter where we run from.
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# --- END PATH FIX ---

import numpy as np
import matplotlib.pyplot as plt

from lab1.bandit_env import TenArmedBanditEnv
from lab1.agents_bandit import EpsilonGreedyAgent


def run_single_episode(env_seed, epsilon, n_steps=2000, n_arms=10):
    """
    Runs one bandit run (fresh bandit with its own reward means),
    using one epsilon-greedy agent for n_steps.
    Returns:
        rewards[t]
        optimal_action_taken[t] (1 if we picked the true best arm at t, else 0)
    """
    # create a fresh bandit with its own q_true means
    env = TenArmedBanditEnv(n_arms=n_arms, seed=env_seed)
    obs, info = env.reset()

    optimal_arm = info["optimal_arm"]

    agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=epsilon, rng_seed=env_seed)

    rewards = np.zeros(n_steps)
    optimal_picks = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        action = agent.select_action()
        obs_next, reward, terminated, truncated, step_info = env.step(action)

        agent.update(action, reward)

        rewards[t] = reward
        optimal_picks[t] = 1 if action == step_info["optimal_arm"] else 0

        obs = obs_next  # not really used in bandit, but keeps structure similar to RL

    return rewards, optimal_picks


def run_experiment(
    epsilons=(0.01, 0.1, 0.2),
    n_runs=1000,
    n_steps=2000,
    n_arms=10,
):
    """
    For each epsilon value:
      - run n_runs independent bandits
      - average reward per timestep across runs
      - average % optimal action across runs
    Returns dict:
      results[epsilon] = {
          "avg_reward": ... shape (n_steps,),
          "optimal_pct": ... shape (n_steps,)
      }
    """
    results = {}

    for eps in epsilons:
        all_rewards = np.zeros((n_runs, n_steps))
        all_optimal = np.zeros((n_runs, n_steps))

        for run_idx in range(n_runs):
            # seed based on run_idx for reproducibility
            rewards, optimal_hits = run_single_episode(
                env_seed=run_idx,
                epsilon=eps,
                n_steps=n_steps,
                n_arms=n_arms,
            )
            all_rewards[run_idx] = rewards
            all_optimal[run_idx] = optimal_hits

        avg_reward = all_rewards.mean(axis=0)  # shape (n_steps,)
        optimal_pct = all_optimal.mean(axis=0) * 100.0  # convert to percentage

        results[eps] = {
            "avg_reward": avg_reward,
            "optimal_pct": optimal_pct,
        }

    return results


def plot_results(results, save_dir="Lab1/figs"):
    """
    Make and save the two required plots:
    1. Average reward over time
    2. % optimal action over time
    """
    # --- Plot 1: Average reward ---
    plt.figure()
    for eps, stats in results.items():
        plt.plot(stats["avg_reward"], label=f"epsilon={eps}")
    plt.xlabel("Timestep")
    plt.ylabel("Average reward")
    plt.title("Average Reward over Time (ε-greedy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_reward_vs_time.png", dpi=200)
    plt.close()

    # --- Plot 2: % Optimal Action ---
    plt.figure()
    for eps, stats in results.items():
        plt.plot(stats["optimal_pct"], label=f"epsilon={eps}")
    plt.xlabel("Timestep")
    plt.ylabel("% Optimal Action")
    plt.title("Percent Optimal Action over Time (ε-greedy)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/optimal_action_vs_time.png", dpi=200)
    plt.close()


if __name__ == "__main__":
    # You can lower n_runs while debugging so it runs fast (like 10 instead of 1000).
    # Final grading should use n_runs=1000, n_steps=2000.
    results = run_experiment(
        epsilons=(0.01, 0.1, 0.2),
        n_runs=1000,
        n_steps=200,
        n_arms=10,
    )

    plot_results(results, save_dir="lab1/figs")

    print("Done. Plots saved in lab1/figs/")
