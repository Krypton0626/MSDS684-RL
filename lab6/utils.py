import os
import random
from typing import List

import gymnasium as gym
import numpy as np
import torch
import matplotlib.pyplot as plt


def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_cartpole_env(seed: int) -> gym.Env:
    """Create a CartPole-v1 env with a fixed seed."""
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """
    Monte Carlo return G_t for each time step (backwards).
    G_t = r_t+1 + gamma * G_{t+1}
    """
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)


def ensure_dir(path: str):
    """Create directory if it does not exist."""
    os.makedirs(path, exist_ok=True)


def plot_with_ci(
    returns_matrix: np.ndarray,
    label: str,
    color: str,
):
    """
    Plot mean episode return with 95% confidence intervals.

    returns_matrix: shape (num_seeds, num_episodes)
    """
    num_seeds, num_episodes = returns_matrix.shape
    x = np.arange(num_episodes)

    mean = returns_matrix.mean(axis=0)
    std = returns_matrix.std(axis=0)
    ci = 1.96 * std / np.sqrt(num_seeds)

    plt.plot(x, mean, label=label, color=color)
    plt.fill_between(x, mean - ci, mean + ci, alpha=0.2, color=color)


def save_returns(path: str, arr: np.ndarray):
    """Save returns array to a .npy file."""
    np.save(path, arr)
