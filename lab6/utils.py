import random
from typing import List

import gymnasium as gym
import numpy as np
import torch


def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def make_cartpole_env(seed: int) -> gym.Env:
    env = gym.make("CartPole-v1")
    env.reset(seed=seed)
    env.action_space.seed(seed)
    return env


def compute_returns(rewards: List[float], gamma: float) -> np.ndarray:
    """Monte Carlo returns G_t computed backwards."""
    G = 0.0
    returns = []
    for r in reversed(rewards):
        G = r + gamma * G
        returns.insert(0, G)
    return np.array(returns, dtype=np.float32)
