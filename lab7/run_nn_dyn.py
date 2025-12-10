import gymnasium as gym
import torch
from neural_dynamics import NeuralDynamicsModel, ReplayBuffer, train_dynamics
from model_based_pg import PolicyNet, reinforce_with_model


def collect_data(env, buffer, steps=5000):
    s, _ = env.reset()
    for _ in range(steps):
        a = env.action_space.sample()
        s_next, r, terminated, truncated, _ = env.step(a)
        buffer.push(s, a, r, s_next)
        if terminated or truncated:
            s,_ = env.reset()
        else:
            s = s_next


def main():
    env = gym.make("MountainCar-v0")

    # Step 1: Collect data
    buffer = ReplayBuffer()
    collect_data(env, buffer, steps=8000)

    # Step 2: Train dynamics model
    model = NeuralDynamicsModel()
    model = train_dynamics(model, buffer, epochs=25)

    # Step 3: Train policy using simulated rollouts
    policy = PolicyNet()
    returns = reinforce_with_model(model, policy, episodes=300)

    print("Model-based PG training complete.")
    print("Final simulated returns:", returns[-10:])


if __name__ == "__main__":
    main()
