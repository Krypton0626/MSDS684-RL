# lab5/mc_sarsa.py
import numpy as np
import gymnasium as gym

from .mc_tilecoder import TileCoder


def _epsilon_greedy(q_values, epsilon, rng: np.random.Generator):
    """
    ε-greedy action selection.
    q_values: 1D array of Q(s,a) for all actions.
    """
    if rng.random() < epsilon:
        return int(rng.integers(len(q_values)))
    return int(np.argmax(q_values))


def semi_gradient_sarsa(
    env_name: str = "MountainCar-v0",
    num_episodes: int = 3000,
    num_tilings: int = 8,
    tiles_per_dim=(8, 8),
    alpha: float = 0.1,
    gamma: float = 1.0,
    epsilon_start: float = 1.0,
    epsilon_end: float = 0.05,
    seed: int = 0,
):
    """
    Semi-gradient SARSA with linear function approximation and tile coding.

    Returns:
        w: np.ndarray, shape (num_actions, num_features)
        episode_lengths: list[int]
        tile_coder: TileCoder instance
    """
    rng = np.random.default_rng(seed)
    env = gym.make(env_name)

    low = env.observation_space.low
    high = env.observation_space.high
    state_bounds = list(zip(low, high))

    tile_coder = TileCoder(
        num_tilings=num_tilings,
        tiles_per_dim=list(tiles_per_dim),
        state_bounds=state_bounds,
    )

    num_actions = env.action_space.n
    num_features = tile_coder.num_features

    # Normalize alpha by num_tilings (important!)
    alpha_eff = alpha / num_tilings

    # Weights: one feature vector per action.
    w = np.zeros((num_actions, num_features), dtype=float)

    episode_lengths: list[int] = []

    def q_values(state):
        """Compute Q(s, :) for all actions using current weights."""
        features = tile_coder.get_features(state)
        return np.dot(w, features)  # shape (num_actions,)

    for episode in range(num_episodes):
        state, _ = env.reset(seed=seed + episode)

        # Linear ε decay across episodes
        frac = episode / max(1, num_episodes - 1)
        epsilon = epsilon_start + frac * (epsilon_end - epsilon_start)

        q_vals = q_values(state)
        action = _epsilon_greedy(q_vals, epsilon, rng)

        done = False
        steps = 0

        while not done and steps < 1000:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

            features = tile_coder.get_features(state)
            q_sa = np.dot(w[action], features)

            if not done:
                q_next_vals = q_values(next_state)
                next_action = _epsilon_greedy(q_next_vals, epsilon, rng)
                td_target = reward + gamma * q_next_vals[next_action]
                td_error = td_target - q_sa

                w[action] += alpha_eff * td_error * features

                state = next_state
                action = next_action
            else:
                # Terminal update
                td_target = reward
                td_error = td_target - q_sa
                w[action] += alpha_eff * td_error * features

        episode_lengths.append(steps)

        if (episode + 1) % 100 == 0:
            avg_len = np.mean(episode_lengths[-100:])
            print(
                f"[{num_tilings} tilings, {tiles_per_dim} tiles] "
                f"Episode {episode + 1}/{num_episodes}, "
                f"avg length (last 100) = {avg_len:.1f}"
            )

    env.close()
    return w, episode_lengths, tile_coder
