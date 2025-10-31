"""
Lab 2 - Dynamic Programming Methods
-----------------------------------
This package provides:
- GridWorld environment (deterministic + stochastic)
- Dynamic Programming algorithms (policy/value iteration)
- Visualization utilities
"""

from .gridworld import GridWorld, GridSpec
from .dp_algorithms import (
    policy_evaluation_sync,
    policy_evaluation_inplace,
    policy_improvement,
    policy_iteration,
    value_iteration,
)
from .visualize import plot_policy, plot_value, plot_delta
