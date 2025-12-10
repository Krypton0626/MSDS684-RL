import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


class NeuralDynamicsModel(nn.Module):
    """
    Predicts next_state (2D) and reward (1D) for MountainCar-v0.
    Input: [state(2), action_onehot(3)] = 5D
    Output: [next_state(2), reward(1)] = 3D
    """

    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 3)
        )

    def forward(self, state, action):
        """
        state: [batch, 2]
        action: [batch] int64
        """
        action_onehot = torch.zeros((state.shape[0], 3), device=state.device)
        action_onehot[torch.arange(state.shape[0]), action] = 1.0

        x = torch.cat([state, action_onehot], dim=1)
        return self.model(x)


class ReplayBuffer:
    def __init__(self, capacity=20000):
        self.capacity = capacity
        self.buffer = []
        self.pos = 0

    def push(self, s, a, r, s_next):
        transition = (s, a, r, s_next)
        if len(self.buffer) < self.capacity:
            self.buffer.append(transition)
        else:
            self.buffer[self.pos] = transition
        self.pos = (self.pos + 1) % self.capacity

    def sample(self, batch_size):
        idx = np.random.choice(len(self.buffer), batch_size, replace=False)
        s,a,r,sn = zip(*[self.buffer[i] for i in idx])
        return (
            torch.tensor(np.array(s), dtype=torch.float32),
            torch.tensor(np.array(a), dtype=torch.long),
            torch.tensor(np.array(r), dtype=torch.float32).unsqueeze(1),
            torch.tensor(np.array(sn), dtype=torch.float32)
        )

    def __len__(self):
        return len(self.buffer)


def train_dynamics(model, buffer, epochs=10, batch_size=256, lr=1e-3):
    """
    Train dynamics model on collected transitions.
    """
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    for epoch in range(epochs):
        if len(buffer) < batch_size:
            continue

        s, a, r, s_next = buffer.sample(batch_size)
        pred = model(s, a)
        target = torch.cat([s_next, r], dim=1)

        loss = loss_fn(pred, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return model
