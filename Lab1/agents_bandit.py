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

    Selects action using:
        a_t = argmax_a [ Q[a] + c * sqrt(ln(t) / N[a]) ]

    where:
        - Q[a]: estimated value of arm a
        - N[a]: number of times arm a was chosen
        - t: current timestep
        - c: exploration parameter
    """

    def __init__(self, n_arms: int, c: float, rng_seed: int | None = None):
        import numpy as np
        self.n_arms = n_arms
        self.c = c
        self.rng = np.random.default_rng(rng_seed)
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms, dtype=int)
        self.t = 0

    def select_action(self):
        self.t += 1

        # Force explore each arm at least once
        untried = np.where(self.N == 0)[0]
        if len(untried) > 0:
            return int(self.rng.choice(untried))

        # UCB score = estimate + confidence bonus
        confidence_bonus = self.c * np.sqrt(np.log(self.t) / self.N)
        ucb_values = self.Q + confidence_bonus
        return int(np.argmax(ucb_values))

    def update(self, action: int, reward: float):
        self.N[action] += 1
        alpha = 1.0 / self.N[action]
        self.Q[action] += alpha * (reward - self.Q[action])
