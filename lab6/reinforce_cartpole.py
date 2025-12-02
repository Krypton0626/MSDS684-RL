import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from networks import PolicyNetwork, ValueNetwork
from utils import set_global_seed, make_cartpole_env, compute_returns


def train_reinforce_single_seed(
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    # TODO: implement vanilla REINFORCE
    raise NotImplementedError


def train_reinforce_baseline_single_seed(
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr_policy: float = 1e-3,
    lr_value: float = 5e-3,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    # TODO: implement REINFORCE with value baseline
    raise NotImplementedError
