import numpy as np
import torch
import torch.optim as optim
from torch.distributions import Categorical

from networks import PolicyNetwork, ValueNetwork
from utils import set_global_seed, make_cartpole_env, compute_returns


def train_reinforce_single_seed(
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr: float = 1e-2,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """
    Vanilla REINFORCE (no baseline) for a single random seed.
    Returns:
        episode_returns: np.array of total reward per episode
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

    for episode in range(num_episodes):
        # Reset env for new episode
        state, _ = env.reset()
        done = False

        log_probs = []
        rewards = []

        while not done:
            # Convert state to tensor on correct device
            state_tensor = torch.from_numpy(state).float().to(device)

            # Forward pass through policy to get logits
            logits = policy(state_tensor)

            # Categorical policy over discrete actions
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)

            # Step environment
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            # Store log prob and reward
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

        # Gradient step
        optimizer.zero_grad()
        policy_loss.backward()
        optimizer.step()

        # Track total return for this episode
        episode_returns.append(sum(rewards))

    env.close()
    return np.array(episode_returns, dtype=np.float32)


def run_episode_reinforce_baseline(env, policy, value_net, device: torch.device):
    """
    Run one full episode using policy + value baseline.

    Returns:
        states:    list of state tensors (on device)
        log_probs: list of log π(a|s)
        rewards:   list of rewards
    """
    state, _ = env.reset()
    done = False

    states = []
    log_probs = []
    rewards = []

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

        state = next_state

    return states, log_probs, rewards


def train_reinforce_baseline_single_seed(
    num_episodes: int = 500,
    gamma: float = 0.99,
    lr_policy: float = 1e-3,
    lr_value: float = 5e-3,
    seed: int = 0,
    device: str = "cpu",
) -> np.ndarray:
    """
    REINFORCE with a learned state-value baseline V(s).

    Actor update uses advantage:
        A_t = G_t - V(s_t)

    Critic is trained with MSE to fit Monte Carlo returns.
    """
    device = torch.device(device)

    # Reproducibility
    set_global_seed(seed)

    # Environment
    env = make_cartpole_env(seed)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    # Networks
    policy = PolicyNetwork(state_dim, action_dim).to(device)
    value_net = ValueNetwork(state_dim).to(device)

    policy_opt = optim.Adam(policy.parameters(), lr=lr_policy)
    value_opt = optim.Adam(value_net.parameters(), lr=lr_value)

    episode_returns = []

    for episode in range(num_episodes):
        # Generate one episode
        states, log_probs, rewards = run_episode_reinforce_baseline(
            env, policy, value_net, device
        )

        # Monte Carlo returns
        returns = compute_returns(rewards, gamma)            # (T,)
        returns_t = torch.from_numpy(returns).to(device)

        # Stack states and get values
        state_batch = torch.stack(states)                    # (T, state_dim)
        values = value_net(state_batch)                      # (T,)

        # Advantage A_t = G_t - V(s_t), but detach baseline for actor
        advantages = returns_t - values.detach()

        # Policy loss
        policy_losses = []
        for log_prob, A_t in zip(log_probs, advantages):
            policy_losses.append(-log_prob * A_t)
        policy_loss = torch.stack(policy_losses).sum()

        # Value loss (MSE between returns and V(s))
        value_loss = torch.nn.functional.mse_loss(values, returns_t)

        # Update actor
        policy_opt.zero_grad()
        policy_loss.backward()
        policy_opt.step()

        # Update critic
        value_opt.zero_grad()
        value_loss.backward()
        value_opt.step()

        episode_returns.append(sum(rewards))

    env.close()
    return np.array(episode_returns, dtype=np.float32)
