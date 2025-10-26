import numpy as np


class TenArmedBanditEnv:
    """
    A 10-armed bandit environment with Gaussian reward distributions.
    Mimics Gymnasium's API for compatibility:
        reset() → (observation, info)
        step(action) → (observation_next, reward, terminated, truncated, info)
    """

    def __init__(self, n_arms: int = 10, reward_std: float = 1.0, seed: int | None = None):
        self.n_arms = n_arms
        self.reward_std = reward_std
        self.rng = np.random.default_rng(seed)

        # True mean reward for each arm
        self.q_true = self.rng.normal(0.0, 1.0, size=n_arms)
        self.current_observation = 0  # dummy state

    def reset(self, seed: int | None = None):
        """Resets RNG (optional) and returns dummy observation + info dict."""
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        info = {
            "q_true": self.q_true.copy(),
            "optimal_arm": int(np.argmax(self.q_true)),
        }
        return self.current_observation, info

    def step(self, action: int):
        """Pull an arm and receive stochastic reward."""
        true_mean = self.q_true[action]
        reward = self.rng.normal(true_mean, self.reward_std)

        observation_next = 0
        terminated = False
        truncated = False
        info = {"optimal_arm": int(np.argmax(self.q_true))}

        return observation_next, reward, terminated, truncated, info
