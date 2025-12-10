"""
runner_lab7.py

Runs all Lab 7 experiments on Taxi-v3 and MountainCar-v0 and saves results
to lab7/data/.

Experiments:
1) Q-learning vs Dyna-Q (Taxi-v3)
2) Dyna-Q vs Dyna-Q+ in a dynamic Taxi environment
3) Uniform Dyna-Q vs Prioritized Sweeping (Taxi-v3)
4) (Optional) Neural dynamics + model-based policy gradient (MountainCar-v0)
"""

from typing import Iterable

import numpy as np
import gymnasium as gym

from .utils_lab7 import make_dirs, DATA_DIR
from .dyna_agents import (
    train_q_learning,
    train_dyna_q,
    train_dyna_q_plus,
    train_prioritized_sweeping,
)
from .dyna_models import TaxiDynamicRewardWrapper
from .viz_lab7 import run_all_plots

# Optional neural dynamics extension (MountainCar-v0, PyTorch)
# These imports will fail gracefully if PyTorch is not installed
try:
    from .neural_dynamics import NeuralDynamicsModel, ReplayBuffer, train_dynamics
    from .model_based_pg import PolicyNet, reinforce_with_model

    HAS_TORCH_EXT = True
except Exception:
    HAS_TORCH_EXT = False


# ---------------------------------------------------------------------
# Helper: Dyna-Q baseline on dynamic Taxi env (no bonuses)
# ---------------------------------------------------------------------


def _train_dyna_q_dynamic(
    num_episodes: int = 500,
    n_planning: int = 10,
    alpha: float = 0.1,
    gamma: float = 0.99,
    epsilon: float = 0.1,
    max_steps_per_episode: int = 200,
    change_step: int = 1000,
    seed: int = 0,
):
    """
    Dyna-Q on a dynamic Taxi environment (reward structure changes
    after `change_step` global steps). No exploration bonuses.
    """
    from .dyna_models import TabularModel
    from .utils_lab7 import epsilon_greedy, set_global_seed

    set_global_seed(seed)

    base_env = gym.make("Taxi-v3")
    env = TaxiDynamicRewardWrapper(base_env, change_step=change_step)
    env.reset(seed=seed)

    n_states = env.observation_space.n
    n_actions = env.action_space.n
    Q = np.zeros((n_states, n_actions), dtype=np.float32)
    model = TabularModel()

    episode_returns = []
    episode_lengths = []

    for ep in range(num_episodes):
        state, _ = env.reset()
        done = False
        total_reward = 0.0
        steps = 0

        while not done and steps < max_steps_per_episode:
            from .utils_lab7 import epsilon_greedy as eps_greedy

            action = eps_greedy(Q, state, n_actions, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-learning update
            if done:
                td_target = reward
            else:
                td_target = reward + gamma * np.max(Q[next_state])
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error

            # model update
            model.update(state, action, reward, next_state)

            # planning
            for _ in range(n_planning):
                sample = model.sample_uniform()
                if sample is None:
                    break
                s_sim, a_sim, r_sim, s_next_sim = sample
                td_target_sim = r_sim + gamma * np.max(Q[s_next_sim])
                td_error_sim = td_target_sim - Q[s_sim, a_sim]
                Q[s_sim, a_sim] += alpha * td_error_sim

            state = next_state
            total_reward += reward
            steps += 1

        episode_returns.append(total_reward)
        episode_lengths.append(steps)

    env.close()
    return Q, np.asarray(episode_returns, dtype=np.float32), np.asarray(
        episode_lengths, dtype=np.int32
    )


# ---------------------------------------------------------------------
# Experiment 1: Q-learning vs Dyna-Q (static Taxi-v3)
# ---------------------------------------------------------------------


def experiment_q_vs_dyna(
    num_episodes: int = 500,
    seeds: Iterable[int] = (0, 1, 2),
):
    """
    Compare:
      - pure Q-learning (n = 0)
      - Dyna-Q with n_planning in {5, 10, 50}
    on standard Taxi-v3.
    """
    make_dirs()

    # -------- Q-learning baseline --------
    all_returns = []
    all_lengths = []
    for seed in seeds:
        _, ep_ret, ep_len = train_q_learning(
            num_episodes=num_episodes,
            seed=seed,
        )
        all_returns.append(ep_ret)
        all_lengths.append(ep_len)

    np.save(DATA_DIR / "q_learning_returns.npy", np.stack(all_returns))
    np.save(DATA_DIR / "q_learning_lengths.npy", np.stack(all_lengths))

    # -------- Dyna-Q with different planning budgets --------
    for n_planning in (5, 10, 50):
        key = f"dyna_q_n{n_planning}"
        all_returns = []
        all_lengths = []
        for seed in seeds:
            _, ep_ret, ep_len = train_dyna_q(
                num_episodes=num_episodes,
                n_planning=n_planning,
                seed=seed,
            )
            all_returns.append(ep_ret)
            all_lengths.append(ep_len)

        np.save(DATA_DIR / f"{key}_returns.npy", np.stack(all_returns))
        np.save(DATA_DIR / f"{key}_lengths.npy", np.stack(all_lengths))

    print("Experiment 1 complete: Q-learning vs Dyna-Q (results saved to lab7/data/).")


# ---------------------------------------------------------------------
# Experiment 2: Dyna-Q vs Dyna-Q+ in dynamic environment
# ---------------------------------------------------------------------


def experiment_dyna_q_plus_dynamic(
    num_episodes: int = 500,
    seeds: Iterable[int] = (0, 1, 2),
    change_step: int = 1000,
):
    """
    Compare Dyna-Q vs Dyna-Q+ on a dynamic Taxi environment with
    a reward change after `change_step` global steps.
    """
    make_dirs()

    # -------- Dyna-Q baseline in dynamic env --------
    all_returns_dq = []
    all_lengths_dq = []
    for seed in seeds:
        _, ep_ret, ep_len = _train_dyna_q_dynamic(
            num_episodes=num_episodes,
            n_planning=10,
            change_step=change_step,
            seed=seed,
        )
        all_returns_dq.append(ep_ret)
        all_lengths_dq.append(ep_len)

    np.save(DATA_DIR / "dyna_q_dynamic_returns.npy", np.stack(all_returns_dq))
    np.save(DATA_DIR / "dyna_q_dynamic_lengths.npy", np.stack(all_lengths_dq))

    # -------- Dyna-Q+ with exploration bonuses --------
    all_returns_dqplus = []
    all_lengths_dqplus = []
    for seed in seeds:
        _, ep_ret, ep_len = train_dyna_q_plus(
            num_episodes=num_episodes,
            n_planning=10,
            kappa=0.01,
            change_step=change_step,
            seed=seed,
        )
        all_returns_dqplus.append(ep_ret)
        all_lengths_dqplus.append(ep_len)

    np.save(
        DATA_DIR / "dyna_q_plus_dynamic_returns.npy",
        np.stack(all_returns_dqplus),
    )
    np.save(
        DATA_DIR / "dyna_q_plus_dynamic_lengths.npy",
        np.stack(all_lengths_dqplus),
    )

    print("Experiment 2 complete: Dyna-Q vs Dyna-Q+ (dynamic env results saved).")


# ---------------------------------------------------------------------
# Experiment 3: Uniform Dyna-Q vs Prioritized Sweeping
# ---------------------------------------------------------------------


def experiment_prioritized_sweeping(
    num_episodes: int = 500,
    seeds: Iterable[int] = (0, 1, 2),
):
    """
    Compare:
      - Dyna-Q with uniform random planning
      - Prioritized Sweeping with same n_planning
    on Taxi-v3.
    """
    make_dirs()

    n_planning = 10

    # -------- Uniform Dyna-Q --------
    all_returns_uniform = []
    all_lengths_uniform = []
    for seed in seeds:
        _, ep_ret, ep_len = train_dyna_q(
            num_episodes=num_episodes,
            n_planning=n_planning,
            seed=seed,
        )
        all_returns_uniform.append(ep_ret)
        all_lengths_uniform.append(ep_len)

    np.save(
        DATA_DIR / "dyna_q_uniform_returns.npy",
        np.stack(all_returns_uniform),
    )
    np.save(
        DATA_DIR / "dyna_q_uniform_lengths.npy",
        np.stack(all_lengths_uniform),
    )

    # -------- Prioritized Sweeping --------
    all_returns_ps = []
    all_lengths_ps = []
    for seed in seeds:
        _, ep_ret, ep_len = train_prioritized_sweeping(
            num_episodes=num_episodes,
            n_planning=n_planning,
            seed=seed,
        )
        all_returns_ps.append(ep_ret)
        all_lengths_ps.append(ep_len)

    np.save(
        DATA_DIR / "prioritized_sweeping_returns.npy",
        np.stack(all_returns_ps),
    )
    np.save(
        DATA_DIR / "prioritized_sweeping_lengths.npy",
        np.stack(all_lengths_ps),
    )

    print("Experiment 3 complete: uniform Dyna-Q vs Prioritized Sweeping saved to lab7/data/.")


# ---------------------------------------------------------------------
# Experiment 4 (Optional): Neural Dynamics + Model-based PG (MountainCar)
# ---------------------------------------------------------------------


def experiment_neural_dynamics_pg(
    collect_steps: int = 8000,
    dyn_epochs: int = 25,
    sim_episodes: int = 300,
):
    """
    Optional extension (requires PyTorch):

    - Collect transitions from real MountainCar-v0 environment
    - Train a neural network dynamics model on (s, a, r, s')
    - Train a policy using REINFORCE on simulated rollouts from the learned model
    - Save simulated returns to lab7/data/ for analysis
    """
    if not HAS_TORCH_EXT:
        print(
            "Skipping neural dynamics experiment: PyTorch or neural modules not available."
        )
        return

    make_dirs()

    env = gym.make("MountainCar-v0")

    # 1) Collect real transitions into replay buffer
    buffer = ReplayBuffer()
    state, _ = env.reset()
    for _ in range(collect_steps):
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, _ = env.step(action)

        buffer.push(state, action, reward, next_state)

        done = terminated or truncated
        if done:
            state, _ = env.reset()
        else:
            state = next_state

    env.close()

    # 2) Train neural dynamics model on collected data
    model = NeuralDynamicsModel()
    model = train_dynamics(
        model,
        buffer,
        epochs=dyn_epochs,
        batch_size=256,
        lr=1e-3,
    )

    # 3) Train policy using simulated rollouts from learned model
    policy = PolicyNet()
    sim_returns = reinforce_with_model(
        model,
        policy,
        episodes=sim_episodes,
        gamma=0.99,
        lr=1e-3,
    )

    np.save(DATA_DIR / "nn_model_based_pg_returns.npy", np.asarray(sim_returns))
    print("Experiment 4 complete: Neural dynamics + model-based PG (results saved).")


# ---------------------------------------------------------------------
# High-level entry point
# ---------------------------------------------------------------------


def run_all_lab7_experiments():
    """
    Run all core Lab 7 experiments and generate plots.

    Experiments:
      1) Q-learning vs Dyna-Q (Taxi-v3)
      2) Dyna-Q vs Dyna-Q+ in dynamic Taxi
      3) Uniform Dyna-Q vs Prioritized Sweeping
      4) (Optional) Neural dynamics + model-based PG (MountainCar-v0)
    """
    experiment_q_vs_dyna()
    experiment_dyna_q_plus_dynamic()
    experiment_prioritized_sweeping()

    # Optional extension experiment (does not affect Taxi results if it fails)
    try:
        experiment_neural_dynamics_pg()
    except Exception as e:
        print("Optional neural dynamics experiment failed:", e)

    # After Taxi-based experiments, generate all plots
    run_all_plots()
    print("All Lab 7 experiments and plots complete.")


if __name__ == "__main__":
    run_all_lab7_experiments()
