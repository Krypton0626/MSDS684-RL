import numpy as np


class EpsilonGreedyAgent:
    """
    ε-greedy bandit agent.
    Chooses random action with probability ε; otherwise exploits argmax(Q).
    Updates Q-values using incremental sample-average update.
    """

    def __init__(self, n_arms: int, epsilon: float, rng_seed: int | None = None):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.rng = np.random.default_rng(rng_seed)
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms, dtype=int)
        self.t = 0

    def select_action(self):
        self.t += 1
        if self.rng.random() < self.epsilon:
            return self.rng.integers(0, self.n_arms)
        return int(np.argmax(self.Q))

    def update(self, action: int, reward: float):
        """Incremental mean update."""
        self.N[action] += 1
        alpha = 1.0 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])
