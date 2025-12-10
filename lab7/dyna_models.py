"""
dyna_models.py

Tabular environment models and wrappers for Lab 7:
- TabularModel: deterministic model (s, a) -> (r, s')
  with predecessor tracking for prioritized sweeping.
- TaxiDynamicRewardWrapper: modifies Taxi-v3 rewards after
  a global step threshold to simulate a changing environment.
"""

from collections import defaultdict
from typing import Dict, Tuple, Optional, Set

import gymnasium as gym


class TabularModel:
    """
    Deterministic tabular model:

        model[(s, a)] = (r, s')

    Also maintains:
      - visited: set of (s, a) pairs ever seen
      - predecessors: mapping s' -> set of (s, a) that lead to s'

    This structure supports:
      - uniform planning (random sampling over visited pairs)
      - prioritized sweeping (querying predecessors of a state)
    """

    def __init__(self) -> None:
        # (s, a) -> (r, s')
        self.model: Dict[Tuple[int, int], Tuple[float, int]] = {}
        # set of (s, a) pairs that have been observed
        self.visited: Set[Tuple[int, int]] = set()
        # s' -> set of (s, a) predecessors
        self.predecessors: Dict[int, Set[Tuple[int, int]]] = defaultdict(set)

    def update(self, state: int, action: int, reward: float, next_state: int) -> None:
        """
        Store / overwrite the observed transition for (state, action).

        If the successor changes, update predecessor links accordingly.
        """
        key = (state, action)

        # If this (s,a) existed before, remove its old predecessor link
        if key in self.model:
            _, old_next = self.model[key]
            if key in self.predecessors[old_next]:
                self.predecessors[old_next].remove(key)

        # Store new transition
        self.model[key] = (reward, next_state)
        self.visited.add(key)
        self.predecessors[next_state].add(key)

    # ---------- Uniform planning helpers ----------

    def sample_uniform(self):
        """
        Sample a random previously observed (s, a, r, s') transition.

        Returns
        -------
        (s, a, r, s') tuple or None if no transitions stored yet.
        """
        if not self.visited:
            return None
        # Convert to list once; small cost for tabular Taxi-v3
        s, a = next(iter(self.visited))  # placeholder, overwritten below

        # Random choice over visited set
        import random
        s, a = random.choice(list(self.visited))
        r, s_next = self.model[(s, a)]
        return s, a, r, s_next

    # ---------- Accessors for prioritized sweeping ----------

    def get_successor(self, state: int, action: int) -> Optional[Tuple[float, int]]:
        """
        Get (reward, next_state) for a given (state, action),
        or None if the transition has never been observed.
        """
        return self.model.get((state, action), None)

    def get_predecessors(self, next_state: int) -> Set[Tuple[int, int]]:
        """
        Get the set of (state, action) pairs that have been observed
        to transition into `next_state`.
        """
        return self.predecessors.get(next_state, set())


class TaxiDynamicRewardWrapper(gym.Wrapper):
    """
    Environment wrapper for Taxi-v3 that *changes the reward structure*
    after a specified number of global environment steps.

    This simulates a non-stationary environment where the learned model
    becomes outdated, which is useful for comparing Dyna-Q vs Dyna-Q+.

    Behavior (after change_step):
      - Successful dropoff reward (default +20) -> new_goal_reward
      - Step penalty (default -1) -> new_step_penalty
      - Illegal pickup/dropoff (-10) is left unchanged.
    """

    def __init__(
        self,
        env: gym.Env,
        change_step: int = 1000,
        new_goal_reward: float = 10.0,
        new_step_penalty: float = -2.0,
    ) -> None:
        super().__init__(env)
        self.change_step = int(change_step)
        self.new_goal_reward = float(new_goal_reward)
        self.new_step_penalty = float(new_step_penalty)

        # Global step counter across episodes
        self.total_steps: int = 0

    def reset(self, **kwargs):
        """
        Standard Gymnasium reset. We intentionally do NOT reset total_steps
        here, because we want the environment change to occur once globally,
        not per-episode.
        """
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Step the underlying Taxi environment, then, if we are past the
        change_step threshold, modify the reward according to the new
        reward structure.
        """
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.total_steps += 1

        # Apply changed reward shaping after threshold
        if self.total_steps >= self.change_step:
            # In Taxi-v3:
            #   +20  -> successful dropoff
            #   -1   -> normal step
            #   -10  -> illegal pickup/dropoff
            if reward == 20:
                reward = self.new_goal_reward
            elif reward == -1:
                reward = self.new_step_penalty
            # leave -10 as is

        return obs, reward, terminated, truncated, info
