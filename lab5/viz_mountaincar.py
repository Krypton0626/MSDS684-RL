# lab5/viz_mountaincar.py
import numpy as np
import matplotlib.pyplot as plt


def plot_learning_curves(config_results, save_path="lab5/figs/learning_curves_all.png"):
    """
    Plot smoothed episode length curves for multiple configurations.

    Args:
        config_results: list of (label, episode_lengths_list)
    """
    plt.figure(figsize=(8, 5))

    for label, lengths in config_results:
        lengths = np.asarray(lengths)
        window = 50

        if len(lengths) >= window:
            kernel = np.ones(window) / window
            smooth = np.convolve(lengths, kernel, mode="valid")
            xs = np.arange(len(smooth)) + window
            plt.plot(xs, smooth, label=label)
        else:
            plt.plot(lengths, label=label)

    plt.xlabel("Episode")
    plt.ylabel("Episode length (steps)")
    plt.title("Semi-Gradient SARSA on MountainCar-v0")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_value_function(w, tile_coder, save_path="lab5/figs/value_heatmap.png"):
    """
    Plot V(s) = max_a Q(s,a) as a heatmap over position × velocity.
    """
    positions = np.linspace(-1.2, 0.6, 60)
    velocities = np.linspace(-0.07, 0.07, 60)

    V = np.zeros((len(positions), len(velocities)))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = np.array([pos, vel])
            features = tile_coder.get_features(state)
            q_vals = np.dot(w, features)
            V[i, j] = np.max(q_vals)

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(
        velocities,
        positions,
        V,
        levels=30,
    )
    plt.colorbar(contour, label="V(s) = max_a Q(s,a)")
    plt.xlabel("Velocity")
    plt.ylabel("Position")
    plt.title("Learned Value Function")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_policy(w, tile_coder, save_path="lab5/figs/policy.png"):
    """
    Plot greedy policy (argmax_a Q(s,a)) over position × velocity.
    """
    positions = np.linspace(-1.2, 0.6, 40)
    velocities = np.linspace(-0.07, 0.07, 40)

    policy = np.zeros((len(positions), len(velocities)), dtype=int)

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            state = np.array([pos, vel])
            features = tile_coder.get_features(state)
            q_vals = np.dot(w, features)
            policy[i, j] = int(np.argmax(q_vals))

    plt.figure(figsize=(8, 5))
    img = plt.imshow(
        policy,
        origin="lower",
        extent=[velocities[0], velocities[-1], positions[0], positions[-1]],
        aspect="auto",
    )
    cbar = plt.colorbar(img, ticks=[0, 1, 2])
    cbar.ax.set_yticklabels(["Left (0)", "No push (1)", "Right (2)"])
    plt.xlabel("Velocity")
    plt.ylabel("Position")
    plt.title("Greedy Policy from Learned Q")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_trajectories(
    w,
    tile_coder,
    env,
    n_episodes=5,
    save_path="lab5/figs/trajectories.png",
):
    """
    Draw a few greedy trajectories on top of the value function.
    """
    positions_list = []
    velocities_list = []

    def greedy_action(state):
        features = tile_coder.get_features(state)
        q_vals = np.dot(w, features)
        return int(np.argmax(q_vals))

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False
        pos_traj = []
        vel_traj = []
        steps = 0

        while not done and steps < 200:
            pos_traj.append(state[0])
            vel_traj.append(state[1])

            action = greedy_action(state)
            state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            steps += 1

        positions_list.append(pos_traj)
        velocities_list.append(vel_traj)

    # Background value function
    positions = np.linspace(-1.2, 0.6, 60)
    velocities = np.linspace(-0.07, 0.07, 60)
    V = np.zeros((len(positions), len(velocities)))

    for i, pos in enumerate(positions):
        for j, vel in enumerate(velocities):
            s = np.array([pos, vel])
            features = tile_coder.get_features(s)
            V[i, j] = np.max(np.dot(w, features))

    plt.figure(figsize=(8, 5))
    contour = plt.contourf(velocities, positions, V, levels=20)
    plt.colorbar(contour, label="V(s)")
    for pos_traj, vel_traj in zip(positions_list, velocities_list):
        plt.plot(vel_traj, pos_traj, linewidth=1.5)

    plt.xlabel("Velocity")
    plt.ylabel("Position")
    plt.title("Sample Greedy Trajectories over Value Function")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
