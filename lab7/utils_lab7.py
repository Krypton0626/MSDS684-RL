"""
utils_lab7.py

Shared utilities for Lab 7:
- directory setup for data/ and figs/
- epsilon-greedy action selection
- seeding helpers
"""

from pathlib import Path
import numpy as np
import random


# Paths relative to this file (lab7/)
LAB7_ROOT = Path(__file__).resolve().parent
DATA_DIR = LAB7_ROOT / "data"
FIGS_DIR = LAB7_ROOT / "figs"


def make_dirs():
    """Create data/ and figs/ directories for Lab 7 if they don't exist."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIGS_DIR.mkdir(parents=True, exist_ok=True)


def set_global_seed(seed: int):
    """
    Set Python and NumPy RNG seeds for reproducibility.
    (Gymnasium envs are seeded separately via env.reset(seed=seed).)
    """
    random.seed(seed)
    np.random.seed(seed)


def epsilon_greedy(Q: np.ndarray, state: int, n_actions: int, epsilon: float) -> int:
    """
    Epsilon-greedy action selection for tabular Q.

    Parameters
    ----------
    Q : np.ndarray
        Q-table with shape [n_states, n_actions].
    state : int
        Current discrete state index.
    n_actions : int
        Number of discrete actions.
    epsilon : float
        Exploration rate in [0, 1].

    Returns
    -------
    int
        Selected action index.
    """
    if np.random.random() < epsilon:
        # explore
        return np.random.randint(n_actions)
    # exploit
    return int(np.argmax(Q[state]))
