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
class UCBAgent:
    """
    Upper Confidence Bound (UCB) bandit agent.

    Action selection rule:
        a_t = argmax_a [ Q[a] + c * sqrt( ln(t) / N[a] ) ]

    Where:
    - Q[a] is the estimated action value of arm a
    - N[a] is how many times we've picked arm a
    - t is the timestep (1-based)
    - c controls exploration strength
    """

    def __init__(self, n_arms: int, c: float, rng_seed: int | None = None):
        self.n_arms = n_arms
        self.c = c
        self.rng = np.random.default_rng(rng_seed)

        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms, dtype=int)
        self.t = 0

    def select_action(self):
        self.t += 1

        # If we haven't tried all arms yet, force-try unvisited arms first
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return int(self.rng.choice(untried))

        # Otherwise use UCB score
        confidence_bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        ucb_score = self.Q + confidence_bonus
        return int(np.argmax(ucb_score))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        alpha = 1.0 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])
