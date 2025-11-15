import numpy as np


def epsilon_greedy_action(Q, state, epsilon, n_actions):
    """
    ε-greedy action selection for a given state.

    With probability ε: choose a random action.
    With probability 1-ε: choose the greedy (argmax) action.
    """
    if np.random.random() < epsilon:
        return np.random.randint(n_actions)
    return int(np.argmax(Q[state]))


def _set_seed(seed):
    """Utility: set NumPy RNG seed if provided."""
    if seed is not None:
        np.random.seed(seed)


def sarsa(env, num_episodes, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    """
    On-policy TD control (SARSA) for CliffWalking-v0.

    Returns
    -------
    Q : np.ndarray
        Learned action-value function of shape [n_states, n_actions].
    episode_returns : np.ndarray
        1D array of length num_episodes with total return per episode.
    """
    _set_seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)

    for ep in range(num_episodes):
        # reset WITHOUT seed each episode to allow randomness across episodes
        state, _ = env.reset()

        action = epsilon_greedy_action(Q, state, epsilon, n_actions)

        done = False
        total_reward = 0.0

        while not done:
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if not done:
                next_action = epsilon_greedy_action(Q, next_state, epsilon, n_actions)
                td_target = reward + gamma * Q[next_state, next_action]
            else:
                # Terminal state: no bootstrap
                td_target = reward

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state
            if not done:
                action = next_action

        episode_returns[ep] = total_reward

    return Q, episode_returns


def q_learning(env, num_episodes, alpha=0.5, gamma=1.0, epsilon=0.1, seed=None):
    """
    Off-policy TD control (Q-learning) for CliffWalking-v0.

    Returns
    -------
    Q : np.ndarray
        Learned action-value function of shape [n_states, n_actions].
    episode_returns : np.ndarray
        1D array of length num_episodes with total return per episode.
    """
    _set_seed(seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions), dtype=np.float64)
    episode_returns = np.zeros(num_episodes, dtype=np.float64)

    for ep in range(num_episodes):
        state, _ = env.reset()

        done = False
        total_reward = 0.0

        while not done:
            action = epsilon_greedy_action(Q, state, epsilon, n_actions)

            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

            if not done:
                td_target = reward + gamma * np.max(Q[next_state])
            else:
                td_target = reward

            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            state = next_state

        episode_returns[ep] = total_reward

    return Q, episode_returns
