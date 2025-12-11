"""
Lab 8 – Modern Deep RL Exploration with PPO on LunarLander-v2.

This script runs PPO with Stable-Baselines3 over a small hyperparameter grid:
- 3 learning rates
- 2 network architectures
- 2 batch sizes (ablation)
- 3 random seeds

Results (evaluation curves + metadata) are saved to lab8/data/ppo_results.npy.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Dict, Tuple, List

import numpy as np
from stable_baselines3 import PPO

from .utils import make_env, evaluate_model, set_global_seeds


ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

ENV_ID = "CartPole-v1"


def train_single_config(
    learning_rate: float,
    net_arch: Tuple[int, int],
    batch_size: int,
    total_timesteps: int,
    eval_freq: int,
    n_eval_episodes: int,
    seed: int,
) -> Dict:
    """
    Train PPO for a single hyperparameter configuration and return logged results.
    """

    print(
        f"Starting run: lr={learning_rate}, net_arch={net_arch}, "
        f"batch_size={batch_size}, seed={seed}"
    )

    set_global_seeds(seed)

    env = make_env(ENV_ID, seed=seed)

    policy_kwargs = dict(net_arch=list(net_arch))

    model = PPO(
        policy="MlpPolicy",
        env=env,
        learning_rate=learning_rate,
        batch_size=batch_size,
        policy_kwargs=policy_kwargs,
        verbose=0,
        seed=seed,
    )

    timesteps: List[int] = []
    eval_means: List[float] = []
    eval_stds: List[float] = []

    timesteps_done = 0
    start_time = time.time()

    while timesteps_done < total_timesteps:
        # Train for eval_freq timesteps, but keep global time.
        model.learn(total_timesteps=eval_freq, reset_num_timesteps=False, progress_bar=False)
        timesteps_done += eval_freq

        mean_r, std_r = evaluate_model(model, env, n_eval_episodes=n_eval_episodes)

        timesteps.append(timesteps_done)
        eval_means.append(mean_r)
        eval_stds.append(std_r)

        print(
            f"[PPO] steps={timesteps_done:7d} "
            f"lr={learning_rate:.1e} arch={net_arch} batch={batch_size} "
            f"eval_mean={mean_r:.2f} +/- {std_r:.2f}"
        )

    training_time = time.time() - start_time
    env.close()

    return {
        "env_id": ENV_ID,
        "learning_rate": learning_rate,
        "net_arch": net_arch,
        "batch_size": batch_size,
        "seed": seed,
        "timesteps": np.array(timesteps, dtype=np.int64),
        "eval_mean_rewards": np.array(eval_means, dtype=np.float32),
        "eval_std_rewards": np.array(eval_stds, dtype=np.float32),
        "training_time_sec": float(training_time),
        "total_timesteps": int(total_timesteps),
        "eval_freq": int(eval_freq),
        "n_eval_episodes": int(n_eval_episodes),
    }


def main():
    """
    Run the full hyperparameter experiment grid and save results.
    """

    # Hyperparameter grid (matches assignment spec)
    learning_rates = [1e-4, 3e-4, 1e-3]
    net_archs = [(64, 64), (256, 256)]
    batch_sizes = [64, 256]  # simple ablation
    seeds = [0, 1, 2]

    total_timesteps = 200_000
    eval_freq = 10_000
    n_eval_episodes = 10

    all_results: List[Dict] = []

    for lr in learning_rates:
        for arch in net_archs:
            for batch in batch_sizes:
                for seed in seeds:
                    result = train_single_config(
                        learning_rate=lr,
                        net_arch=arch,
                        batch_size=batch,
                        total_timesteps=total_timesteps,
                        eval_freq=eval_freq,
                        n_eval_episodes=n_eval_episodes,
                        seed=seed,
                    )
                    all_results.append(result)

    out_path = DATA_DIR / "ppo_results.npy"
    np.save(out_path, np.array(all_results, dtype=object))
    print(f"\nSaved {len(all_results)} result dicts to {out_path}")


if __name__ == "__main__":
    main()
