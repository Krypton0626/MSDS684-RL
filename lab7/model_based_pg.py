import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class PolicyNet(nn.Module):
    """
    Simple 2-layer MLP for MountainCar policy.
    """
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, x):
        return torch.softmax(self.net(x), dim=1)


def simulate_episode(model, policy, max_steps=200):
    """
    Generate a fake trajectory using ONLY the learned dynamics model.
    """
    s = torch.tensor([[-0.5, 0.0]], dtype=torch.float32)  # start state approx.
    log_probs = []
    rewards = []

    for _ in range(max_steps):
        probs = policy(s)
        m = torch.distributions.Categorical(probs)
        a = m.sample()

        log_probs.append(m.log_prob(a))

        pred = model(s, a)  # predicted next state + reward
        next_state = pred[0, :2].unsqueeze(0)
        reward = pred[0, 2].item()

        rewards.append(reward)

        s = next_state

    return log_probs, rewards


def reinforce_with_model(model, policy, episodes=200, gamma=0.99, lr=1e-3):
    optimizer = optim.Adam(policy.parameters(), lr=lr)

    returns_history = []

    for _ in range(episodes):
        log_probs, rewards = simulate_episode(model, policy)
        returns_history.append(sum(rewards))

        # Compute discounted returns
        G = 0
        discounts = []
        for r in reversed(rewards):
            G = r + gamma * G
            discounts.insert(0, G)

        discounts = torch.tensor(discounts, dtype=torch.float32)

        # Policy gradient update
        loss = 0
        for log_p, Gt in zip(log_probs, discounts):
            loss -= log_p * Gt

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return returns_history
