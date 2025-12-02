# lab6/reinforce_cartpole.py

import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

# Support both:
# - running as part of the lab6 package (from repo root)
# - running as standalone scripts inside the lab6 folder
try:
    # Package-style imports (when called via run_lab6.py at repo root)
    from .networks import PolicyNetwork, ValueNetwork
    from .utils import set_global_seed, make_cartpole_env, compute_returns
except ImportError:
    # Script-style imports (when running inside lab6/)
    from networks import PolicyNetwork, ValueNetwork
    from utils import set_global_seed, make_cartpole_env, compute_returns


def _dist_entropy(dist: Categorical) -> float:
    """Utility to get scalar entropy from a Categorical distribution."""
    # dist.entropy() returns a tensor; convert to Python float
    return dist.entropy().item()


def train_reinforce_single_seed(
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Vanilla REINFORCE (no baseline) for a single random seed.

    Returns:
        episode_returns:  np.array of total reward per episode  (shape: [num_episodes])
        episode_entropies: np.array of mean policy entropy per episode
    """
    device = torch.device(device)

    # Reproducibility
    set_global_seed(seed)

    # Environment
    env = make_cartpole_env(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Policy network + optimizer
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    episode_returns = []
    episode_entropies = []

    for episode in range(num_episodes):
        state, _ = env.reset()
        done = False

        log_probs = []
        rewards = []
        entropies = []

        while not done:
            state_tensor = torch.from_numpy(state).float().to(device)

            logits = policy(state_tensor)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Track entropy for this step
            entropies.append(_dist_entropy(dist))

            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            log_probs.append(log_prob)
            rewards.append(reward)
            state = next_state

        # Monte Carlo returns G_t
        returns = compute_returns(rewards, gamma)              # shape (T,)
        returns_t = torch.from_numpy(returns).to(device)       # tensor on device

        # Policy gradient loss: sum_t -log π(a_t|s_t) * G_t
        policy_loss = []
        for log_prob, G_t in zip(log_probs, returns_t):
            policy_loss.append(-log_prob * G_t)
        policy_loss = torch.stack(policy_loss).sum()

        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Per-episode stats
        episode_returns.append(sum(rewards))
        episode_entropies.append(float(np.mean(entropies)))

    env.close()
    return (
        np.array(episode_returns, dtype=np.float32),
        np.array(episode_entropies, dtype=np.float32),
    )


def run_episode_reinforce_baseline(env, policy, value_net, device: torch.device):
    """
    Run one full episode using policy + value baseline.

    Returns:
        states:    list of state tensors (on device)
        log_probs: list of log π(a|s)
        rewards:   list of rewards
        entropies: list of per-step entropies
    """
    state, _ = env.reset()
    done = False

    states = []
    log_probs = []
    rewards = []
    entropies = []

    while not done:
        state_tensor = torch.from_numpy(state).float().to(device)

        logits = policy(state_tensor)
        dist = Categorical(logits=logits)
        action = dist.sample()
        log_prob = dist.log_prob(action)

        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated

        states.append(state_tensor)
        log_probs.append(log_prob)
        rewards.append(reward)
        entropies.append(_dist_entropy(dist))

        state = next_state

    return states, log_probs, rewards, entropies


def train_reinforce_baseline_single_seed(
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr_policy: float = 1e-3,
    lr_value: float = 5e-3,
    seed: int = 0,
    device: str = "cpu",
) -> tuple[np.ndarray, np.ndarray]:
    """
    REINFORCE with a learned state-value baseline V(s).

    Actor update uses advantage:
        A_t = G_t - V(s_t)

    Critic is trained with MSE to fit Monte Carlo returns.

    Returns:
        episode_returns:   np.array [num_episodes]
        episode_entropies: np.array [num_episodes]
    """
    device = torch.device(device)

    set_global_seed(seed)

    env = make_cartpole_env(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    policy = PolicyNetwork(state_dim, action_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)

    policy_opt = optim.Adam(policy.parameters(), lr=lr_policy)
    value_opt = optim.Adam(value_net.parameters(), lr=lr_value)

    episode_returns = []
    episode_entropies = []

    for episode in range(num_episodes):
        states, log_probs, rewards, entropies = run_episode_reinforce_baseline(
            env, policy, value_net, device
        )

        returns = compute_returns(rewards, gamma)            # (T,)
        returns_t = torch.from_numpy(returns).to(device)

        state_batch = torch.stack(states)                    # (T, state_dim)
        values = value_net(state_batch)                      # (T,)

        advantages = returns_t - values.detach()

        policy_losses = []
        for log_prob, A_t in zip(log_probs, advantages):
            policy_losses.append(-log_prob * A_t)
        policy_loss = torch.stack(policy_losses).sum()

        value_loss = torch.nn.functional.mse_loss(values, returns_t)

        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        episode_returns.append(sum(rewards))
        episode_entropies.append(float(np.mean(entropies)))

    env.close()
    return (
        np.array(episode_returns, dtype=np.float32),
        np.array(episode_entropies, dtype=np.float32),
    )


def run_multi_seed(
    algo: str,
    num_seeds: int = 10,
    num_episodes: int = 500,
    gamma: float = 0.99,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run either 'reinforce' or 'reinforce_baseline' for multiple seeds.

    Returns:
        returns_matrix:  np.ndarray of shape (num_seeds, num_episodes)
        entropy_matrix:  np.ndarray of shape (num_seeds, num_episodes)
    """
    all_returns = []
    all_entropies = []

    for seed in range(num_seeds):
        if algo == "reinforce":
            ep_returns, ep_entropies = train_reinforce_single_seed(
                num_episodes=num_episodes,
                gamma=gamma,
                seed=seed,
            )
        elif algo == "reinforce_baseline":
            ep_returns, ep_entropies = train_reinforce_baseline_single_seed(
                num_episodes=num_episodes,
                gamma=gamma,
                seed=seed,
            )
        else:
            raise ValueError(f"Unknown algo: {algo}")

        all_returns.append(ep_returns)
        all_entropies.append(ep_entropies)

    return (
        np.stack(all_returns, axis=0),
        np.stack(all_entropies, axis=0),
    )
