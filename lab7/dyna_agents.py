"""
dyna_agents.py

Core tabular agents for Lab 7 on Taxi-v3:

- train_q_learning:        pure model-free Q-learning (baseline)
- train_dyna_q:            Dyna-Q with uniform random planning
- train_dyna_q_plus:       Dyna-Q+ with exploration bonuses in planning
- train_prioritized_sweeping: prioritized sweeping with a tabular model
"""

from typing import Tuple

import gymnasium as gym
import numpy as np

from .utils_lab7 import epsilon_greedy, set_global_seed
from .dyna_models import TabularModel, TaxiDynamicRewardWrapper


# ---------------------------------------------------------------------
# Helper to create Taxi env (optionally wrapped)
# ---------------------------------------------------------------------


def _make_taxi_env(seed: int = 0) -> gym.Env:
    """Create a plain Taxi-v3 environment with seeding."""
    env = gym.make("Taxi-v3")
    env.reset(seed=seed)
    return env


# ---------------------------------------------------------------------
# 1. Pure Q-learning (baseline, model-free)
# ---------------------------------------------------------------------


def train_q_learning(
    num_episodes: int = 500,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    max_steps_per_episode: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Tabular Q-learning on Taxi-v3 (no planning).

    Returns
    -------
    Q : np.ndarray
        Learned Q-table with shape [n_states, n_actions].
    episode_returns : np.ndarray
        Episodic returns, shape [num_episodes].
    episode_lengths : np.ndarray
        Number of steps per episode, shape [num_episodes].
    """
    set_global_seed(seed)
    env = _make_taxi_env(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            action = epsilon_greedy(Q, state, n_actions, epsilon)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state])

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    env.close()
    return Q, np.asarray(episode_returns, dtype=np.float32), np.asarray(
        episode_lengths, dtype=np.int32
    )


# ---------------------------------------------------------------------
# 2. Dyna-Q with uniform planning
# ---------------------------------------------------------------------


def train_dyna_q(
    num_episodes: int = 500,
    n_planning: int = 5,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    max_steps_per_episode: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dyna-Q on Taxi-v3 with a tabular model and uniform random planning.

    Parameters
    ----------
    n_planning : int
        Number of simulated (model-based) updates per real environment step.

    Returns
    -------
    Q, episode_returns, episode_lengths
    """
    set_global_seed(seed)
    env = _make_taxi_env(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    model = TabularModel()

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # ---------- 1. Direct RL (real step) ----------
            action = epsilon_greedy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state])

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # ---------- 2. Model learning ----------
            model.update(state, action, reward, next_state)

            # ---------- 3. Planning (uniform sampling) ----------
            for _ in range(n_planning):
                sample = model.sample_uniform()
                if sample is None:
                    break
                s_sim, a_sim, r_sim, s_next_sim = sample

                td_target_sim = r_sim + gamma * np.max(Q[s_next_sim])
                td_error_sim = td_target_sim - Q[s_sim, a_sim]
                Q[s_sim, a_sim] += alpha * td_error_sim

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    env.close()
    return Q, np.asarray(episode_returns, dtype=np.float32), np.asarray(
        episode_lengths, dtype=np.int32
    )


# ---------------------------------------------------------------------
# 3. Dyna-Q+ with exploration bonuses in planning
# ---------------------------------------------------------------------


def train_dyna_q_plus(
    num_episodes: int = 500,
    n_planning: int = 10,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    kappa: float = 0.01,
    max_steps_per_episode: int = 200,
    change_step: int = 1000,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Dyna-Q+ on a dynamic Taxi environment.

    Uses TaxiDynamicRewardWrapper to change the reward structure after
    `change_step` global steps. During planning, an exploration bonus
    kappa * sqrt(tau) is added to the reward, where tau is the time since
    the last real visit of (s, a).
    """
    set_global_seed(seed)

    base_env = gym.make("Taxi-v3")
    env = TaxiDynamicRewardWrapper(base_env, change_step=change_step)
    env.reset(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    model = TabularModel()

    episode_returns = []
    episode_lengths = []

    # Global time index for Dyna-Q+
    t = 0
    last_visited = {}  # (s, a) -> time of last REAL visit

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            t += 1

            # ---------- 1. Direct RL ----------
            action = epsilon_greedy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state])

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # ---------- 2. Model learning ----------
            model.update(state, action, reward, next_state)

            # Track real visit time
            last_visited[(state, action)] = t

            # ---------- 3. Planning with exploration bonus ----------
            for _ in range(n_planning):
                sample = model.sample_uniform()
                if sample is None:
                    break
                s_sim, a_sim, r_sim, s_next_sim = sample

                # Time since last real visit of (s_sim, a_sim)
                tau = t - last_visited.get((s_sim, a_sim), 0)
                r_bonus = r_sim + kappa * np.sqrt(float(tau))

                td_target_sim = r_bonus + gamma * np.max(Q[s_next_sim])
                td_error_sim = td_target_sim - Q[s_sim, a_sim]
                Q[s_sim, a_sim] += alpha * td_error_sim

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    env.close()
    return Q, np.asarray(episode_returns, dtype=np.float32), np.asarray(
        episode_lengths, dtype=np.int32
    )


# ---------------------------------------------------------------------
# 4. Prioritized Sweeping
# ---------------------------------------------------------------------


def train_prioritized_sweeping(
    num_episodes: int = 500,
    n_planning: int = 10,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    theta: float = 0.01,
    max_steps_per_episode: int = 200,
    seed: int = 0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prioritized Sweeping on Taxi-v3.

    Uses a priority queue (heapq) keyed by |TD error| to focus planning
    updates on the most "surprising" transitions and their predecessors.
    """
    import heapq

    set_global_seed(seed)
    env = _make_taxi_env(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)

    model = TabularModel()

    # Priority queue of (-priority, state, action)
    # Using negative because heapq is a min-heap.
    pq = []

    def push_to_queue(priority: float, s: int, a: int):
        if priority > theta:
            heapq.heappush(pq, (-priority, s, a))

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            # ---------- 1. Direct RL ----------
            action = epsilon_greedy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Update model with this real transition
            model.update(state, action, reward, next_state)

            # Compute TD error and update Q
            best_next = 0.0 if done else float(np.max(Q[next_state]))
            td_target = reward + gamma * best_next
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # Priority for this (s, a)
            push_to_queue(abs(td_error), state, action)

            # ---------- 2. Planning steps ----------
            for _ in range(n_planning):
                if not pq:
                    break

                neg_p, s_p, a_p = heapq.heappop(pq)
                # Skip if we no longer have a model entry
                succ = model.get_successor(s_p, a_p)
                if succ is None:
                    continue
                r_p, s_next_p = succ

                best_next_p = float(np.max(Q[s_next_p]))
                td_target_p = r_p + gamma * best_next_p
                td_error_p = td_target_p - Q[s_p, a_p]
                Q[s_p, a_p] += alpha * td_error_p

                # Add predecessors of s_p into the queue
                for (s_pred, a_pred) in model.get_predecessors(s_p):
                    succ_pred = model.get_successor(s_pred, a_pred)
                    if succ_pred is None:
                        continue
                    r_pred, s_next_pred = succ_pred
                    best_next_pred = float(np.max(Q[s_next_pred]))
                    td_target_pred = r_pred + gamma * best_next_pred
                    td_error_pred = td_target_pred - Q[s_pred, a_pred]
                    push_to_queue(abs(td_error_pred), s_pred, a_pred)

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    env.close()
    return Q, np.asarray(episode_returns, dtype=np.float32), np.asarray(
        episode_lengths, dtype=np.int32
    )
