"""
Utility functions for Lab 8 PPO experiments.
"""

from __future__ import annotations

import os
import random
from typing import Dict, Iterable, List, Tuple

import gymnasium as gym
import numpy as np
import torch
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor


def set_global_seeds(seed: int) -> None:
    """
    Set seeds for Python, NumPy, and PyTorch to improve reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    # For GPU runs, this is still helpful if CUDA is available.
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_env(env_id: str = "LunarLander-v2", seed: int = 0):
    """
    Create a monitored Gymnasium environment with a given seed.
    """
    env = gym.make(env_id)
    # Gymnasium reset signature: obs, info
    env.reset(seed=seed)
    env.action_space.seed(seed)
    env = Monitor(env)
    return env


def evaluate_model(model, env, n_eval_episodes: int = 10) -> Tuple[float, float]:
    """
    Wrapper around Stable-Baselines3's evaluate_policy for convenience.

    Returns:
        mean_reward, std_reward over n_eval_episodes.
    """
    mean_reward, std_reward = evaluate_policy(
        model,
        env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True,
    )
    return float(mean_reward), float(std_reward)


def group_by_keys(
    results: Iterable[Dict],
    keys: Tuple[str, ...],
) -> Dict[Tuple, List[Dict]]:
    """
    Group a list of result dicts by the specified key fields.

    Args:
        results: iterable of dictionaries with fields like 'learning_rate', 'net_arch', etc.
        keys: tuple of keys to group by.

    Returns:
        dict mapping (key1_value, key2_value, ...) -> list of result dicts
    """
    grouped: Dict[Tuple, List[Dict]] = {}
    for r in results:
        key = tuple(r[k] for k in keys)
        if key not in grouped:
            grouped[key] = []
        grouped[key].append(r)
    return grouped
