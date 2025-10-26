import gymnasium as gym
import numpy as np

def explore_env(env_name, episodes=100):
    env = gym.make(env_name)
    total_rewards = []
    for ep in range(episodes):
        obs, info = env.reset(seed=42)
        done = False
        total_reward = 0
        while not done:
            action = env.action_space.sample()  # random policy
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            done = terminated or truncated
        total_rewards.append(total_reward)
    env.close()
    print(f"\nEnvironment: {env_name}")
    print("Observation space:", env.observation_space)
    print("Action space:", env.action_space)
    print(f"Average total reward over {episodes} episodes: {np.mean(total_rewards):.3f}")
    return np.mean(total_rewards)

if __name__ == "__main__":
    explore_env("FrozenLake-v1")
    explore_env("Taxi-v3")

if __name__ == "__main__":
    avg1 = explore_env("FrozenLake-v1")
    avg2 = explore_env("Taxi-v3")
    print("\nSummary:")
    print(f"FrozenLake-v1 avg reward: {avg1:.3f}")
    print(f"Taxi-v3 avg reward: {avg2:.3f}")