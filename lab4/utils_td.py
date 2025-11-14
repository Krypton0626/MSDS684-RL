import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import gymnasium as gym


def compute_mean_and_ci(returns_2d):
    """
    Compute mean and 95% confidence interval over seeds.

    Parameters
    ----------
    returns_2d : np.ndarray
        Shape [num_seeds, num_episodes].

    Returns
    -------
    mean : np.ndarray
    lower : np.ndarray
    upper : np.ndarray
    """
    mean = returns_2d.mean(axis=0)
    stderr = returns_2d.std(axis=0, ddof=1) / np.sqrt(returns_2d.shape[0])
    ci = 1.96 * stderr
    lower = mean - ci
    upper = mean + ci
    return mean, lower, upper


def plot_learning_curves(
    sarsa_returns,
    qlearning_returns,
    out_path,
    title="CliffWalking: SARSA vs Q-learning",
):
    """
    Plot mean episode returns with 95% CIs for SARSA and Q-learning.

    Parameters
    ----------
    sarsa_returns : np.ndarray
        Shape [num_seeds, num_episodes].
    qlearning_returns : np.ndarray
        Shape [num_seeds, num_episodes].
    out_path : str or Path
        Where to save the figure.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    mean_sarsa, low_sarsa, up_sarsa = compute_mean_and_ci(sarsa_returns)
    mean_q, low_q, up_q = compute_mean_and_ci(qlearning_returns)

    episodes = np.arange(1, mean_sarsa.shape[0] + 1)

    plt.figure(figsize=(10, 5))
    # SARSA
    plt.plot(episodes, mean_sarsa, label="SARSA")
    plt.fill_between(episodes, low_sarsa, up_sarsa, alpha=0.2)
    # Q-learning
    plt.plot(episodes, mean_q, label="Q-learning")
    plt.fill_between(episodes, low_q, up_q, alpha=0.2)

    plt.xlabel("Episode")
    plt.ylabel("Sum of rewards per episode")
    plt.title(title)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_value_heatmap(Q, shape, out_path, title="State-value function V(s)"):
    """
    Visualize V(s) = max_a Q(s,a) as a heatmap.

    Parameters
    ----------
    Q : np.ndarray
        [n_states, n_actions].
    shape : tuple
        Grid shape, e.g. (4, 12) for CliffWalking.
    out_path : str or Path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    V = np.max(Q, axis=1).reshape(shape)

    plt.figure(figsize=(12, 3))
    im = plt.imshow(V, origin="upper", interpolation="nearest")
    plt.colorbar(im, label="V(s)")
    plt.title(title)
    plt.xticks(range(shape[1]))
    plt.yticks(range(shape[0]))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_policy_arrows(Q, shape, out_path, title="Greedy policy π(s)"):
    """
    Visualize a greedy policy as arrow symbols on the grid.

    Action mapping assumed:
    0: Up, 1: Right, 2: Down, 3: Left (Gym CliffWalking default).
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    n_states, n_actions = Q.shape
    assert n_states == shape[0] * shape[1], "Shape does not match Q."

    # Arrow mapping for actions
    arrow_map = {0: "↑", 1: "→", 2: "↓", 3: "←"}

    V = np.max(Q, axis=1).reshape(shape)
    greedy_actions = np.argmax(Q, axis=1).reshape(shape)

    plt.figure(figsize=(12, 3))
    im = plt.imshow(V, origin="upper", interpolation="nearest")
    plt.colorbar(im, label="V(s)")
    plt.title(title)

    for r in range(shape[0]):
        for c in range(shape[1]):
            s = r * shape[1] + c
            a = greedy_actions[r, c]
            plt.text(
                c,
                r,
                arrow_map.get(int(a), "?"),
                ha="center",
                va="center",
                fontsize=10,
                color="black",
            )

    plt.xticks(range(shape[1]))
    plt.yticks(range(shape[0]))
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()


def plot_greedy_trajectory(env_name, Q, max_steps, out_path, title="Sample trajectory"):
    """
    Roll out a greedy policy from the start state and plot the path on the grid.

    Parameters
    ----------
    env_name : str
        e.g. "CliffWalking-v0".
    Q : np.ndarray
        Learned Q-table.
    max_steps : int
        Safety cap on episode length.
    out_path : str or Path
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    env = gym.make(env_name)
    n_cols = 12  # specific to CliffWalking
    n_rows = 4

    state, _ = env.reset()
    done = False

    states = [state]

    for _ in range(max_steps):
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, _ = env.step(action)
        states.append(next_state)
        done = terminated or truncated
        if done:
            break
        state = next_state

    env.close()

    # Convert flat states to (row, col)
    rows = [s // n_cols for s in states]
    cols = [s % n_cols for s in states]

    # Background grid
    plt.figure(figsize=(12, 3))
    plt.imshow(
        np.zeros((n_rows, n_cols)),
        origin="upper",
        interpolation="nearest",
        cmap="gray_r",
    )
    plt.plot(cols, rows, marker="o")
    plt.scatter(cols[0], rows[0], marker="s", s=80, label="Start")
    plt.scatter(cols[-1], rows[-1], marker="*", s=120, label="End")
    plt.title(title)
    plt.xticks(range(n_cols))
    plt.yticks(range(n_rows))
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches="tight")
    plt.close()
