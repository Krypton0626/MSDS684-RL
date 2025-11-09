import os
import csv
import argparse
from pathlib import Path

import numpy as np
import gymnasium as gym
from collections import defaultdict

from utils_mc import (
    compute_returns, first_visit_indices, epsilon_greedy_action,
    init_q_default, to_state_key
)

def train_mc_control_blackjack(
    num_episodes=500_000,
    epsilon_start=1.0,
    epsilon_min=0.02,
    epsilon_decay=0.99995,
    gamma=1.0,
    seed=7,
    log_every=50_000
):
    """
    First-visit on-policy Monte Carlo control with ε-soft policies for Blackjack-v1.
    - Tabular Q with incremental average updates (alpha = 1/N)
    - ε decays multiplicatively to epsilon_min
    """
    np.random.seed(seed)
    env = gym.make("Blackjack-v1", sab=False)  # standard (non-S&B simplified variant)
    nA = env.action_space.n  # 0=stick, 1=hit

    Q = init_q_default(nA)
    returns_count = defaultdict(lambda: np.zeros(nA, dtype=int))
    episode_returns = []

    epsilon = float(epsilon_start)

    for ep in range(1, num_episodes + 1):
        obs, _ = env.reset()
        states, actions, rewards = [], [], []
        done = False

        # Generate one complete episode
        while not done:
            s = to_state_key(obs)
            a = epsilon_greedy_action(Q, s, epsilon, nA)
            next_obs, r, terminated, truncated, _ = env.step(a)

            states.append(s)
            actions.append(a)
            rewards.append(r)

            obs = next_obs
            done = terminated or truncated

        # Backward returns
        Gs = compute_returns(rewards, gamma=gamma)

        # First-visit updates
        fv_idxs = first_visit_indices(states)
        visited = set()
        for t in range(len(states)):
            if t not in fv_idxs:
                continue
            s, a, G = states[t], actions[t], Gs[t]
            if (s, a) in visited:
                continue
            visited.add((s, a))
            returns_count[s][a] += 1
            alpha = 1.0 / returns_count[s][a]
            Q[s][a] += alpha * (G - Q[s][a])

        episode_returns.append(sum(rewards))

        # ε decay
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        if log_every and ep % log_every == 0:
            window = episode_returns[-10_000:] if len(episode_returns) >= 10_000 else episode_returns
            print(f"[ep {ep}] ε={epsilon:.4f} | avg_return(last {len(window)})={np.mean(window):.3f}")

    env.close()
    return Q, np.array(episode_returns, dtype=float)

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="First-visit on-policy MC control for Blackjack-v1")
    p.add_argument("--episodes", type=int, default=500_000)
    p.add_argument("--epsilon-start", type=float, default=1.0)
    p.add_argument("--epsilon-min", type=float, default=0.02)
    p.add_argument("--epsilon-decay", type=float, default=0.99995)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=50_000)
    p.add_argument("--tag", type=str, default="", help="optional run tag suffix for filenames")
    args = p.parse_args()

    tag = f"_{args.tag}" if args.tag else ""

    Q, returns = train_mc_control_blackjack(
        num_episodes=args.episodes,
        epsilon_start=args.epsilon_start,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        gamma=args.gamma,
        seed=args.seed,
        log_every=args.log_every
    )

    Path("data").mkdir(exist_ok=True)
    Path("figs").mkdir(exist_ok=True)

    # Save artifacts with tag
    np.save(f"data/Q_blackjack{tag}.npy", dict(Q), allow_pickle=True)
    np.save(f"data/episode_returns{tag}.npy", returns)

    # CSV for iteration narrative
    with open(f"data/episode_returns{tag}.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["episode", "return"])
        for i, r in enumerate(returns, 1):
            w.writerow([i, r])

    print(f"Saved: data/Q_blackjack{tag}.npy, data/episode_returns{tag}.npy/.csv")
