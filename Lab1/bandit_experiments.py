import os
import sys

# --- PATH FIX so imports work no matter where we run from ---
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt

from lab1.bandit_env import TenArmedBanditEnv
from lab1.agents_bandit import EpsilonGreedyAgent, UCBAgent


def run_single_episode(agent_type, agent_param, env_seed, n_steps=2000, n_arms=10):
    """
    agent_type: "egreedy" or "ucb"
    agent_param:
        - if 'egreedy', this is epsilon (float)
        - if 'ucb', this is c (float)
    """
    env = TenArmedBanditEnv(n_arms=n_arms, seed=env_seed)
    obs, info = env.reset()
    optimal_arm = info["optimal_arm"]

    if agent_type == "egreedy":
        agent = EpsilonGreedyAgent(n_arms=n_arms, epsilon=agent_param, rng_seed=env_seed)
        label = f"epsilon={agent_param}"
    elif agent_type == "ucb":
        agent = UCBAgent(n_arms=n_arms, c=agent_param, rng_seed=env_seed)
        label = f"ucb_c={agent_param}"
    else:
        raise ValueError("agent_type must be 'egreedy' or 'ucb'")

    rewards = np.zeros(n_steps)
    optimal_picks = np.zeros(n_steps, dtype=int)

    for t in range(n_steps):
        action = agent.select_action()
        obs_next, reward, terminated, truncated, step_info = env.step(action)
        agent.update(action, reward)

        rewards[t] = reward
        optimal_picks[t] = 1 if action == step_info["optimal_arm"] else 0

        obs = obs_next

    return rewards, optimal_picks, label


def run_experiment(
    egreedy_epsilons=(0.01, 0.1, 0.2),
    ucb_cs=(1.0, 2.0),
    n_runs=1000,
    n_steps=2000,
    n_arms=10,
):
    """
    Runs epsilon-greedy and UCB across many independent bandit problems.
    Returns:
        results[label] = {
            "avg_reward": (n_steps,),
            "optimal_pct": (n_steps,)
        }
    label examples:
        "epsilon=0.1"
        "ucb_c=2.0"
    """
    results = {}

    # Sweep epsilon-greedy variants
    for eps in egreedy_epsilons:
        all_rewards = np.zeros((n_runs, n_steps))
        all_optimal = np.zeros((n_runs, n_steps))

        for run_idx in range(n_runs):
            rewards, optimal_hits, label = run_single_episode(
                agent_type="egreedy",
                agent_param=eps,
                env_seed=run_idx,
                n_steps=n_steps,
                n_arms=n_arms,
            )
            all_rewards[run_idx] = rewards
            all_optimal[run_idx] = optimal_hits

        results[label] = {
            "avg_reward": all_rewards.mean(axis=0),
            "optimal_pct": all_optimal.mean(axis=0) * 100.0,
        }

    # Sweep UCB variants
    for c_val in ucb_cs:
        all_rewards = np.zeros((n_runs, n_steps))
        all_optimal = np.zeros((n_runs, n_steps))

        for run_idx in range(n_runs):
            rewards, optimal_hits, label = run_single_episode(
                agent_type="ucb",
                agent_param=c_val,
                env_seed=run_idx,
                n_steps=n_steps,
                n_arms=n_arms,
            )
            all_rewards[run_idx] = rewards
            all_optimal[run_idx] = optimal_hits

        results[label] = {
            "avg_reward": all_rewards.mean(axis=0),
            "optimal_pct": all_optimal.mean(axis=0) * 100.0,
        }

    return results


def plot_results(results, save_dir="lab1/figs"):
    # Plot 1: Average reward
    plt.figure()
    for label, stats in results.items():
        plt.plot(stats["avg_reward"], label=label)
    plt.xlabel("Timestep")
    plt.ylabel("Average reward")
    plt.title("Average Reward over Time (ε-greedy vs UCB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/avg_reward_vs_time.png", dpi=200)
    plt.close()

    # Plot 2: % Optimal Action
    plt.figure()
    for label, stats in results.items():
        plt.plot(stats["optimal_pct"], label=label)
    plt.xlabel("Timestep")
    plt.ylabel("% Optimal Action")
    plt.title("% Optimal Action Over Time (ε-greedy vs UCB)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{save_dir}/optimal_action_vs_time.png", dpi=200)
    plt.close()
