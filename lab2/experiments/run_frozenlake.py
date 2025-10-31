import os
import gymnasium as gym
from src.dp_algorithms import policy_iteration, value_iteration
from src.visualize import plot_value_heatmap, plot_policy_arrows, plot_delta_curve


def ensure_results():
    os.makedirs("results", exist_ok=True)


def solve_frozenlake(map_name="4x4", slippery=True, gamma=0.99, theta=1e-7, tag="frozenlake"):
    env = gym.make("FrozenLake-v1", is_slippery=slippery, map_name=map_name)
    P = env.unwrapped.P
    nS = env.observation_space.n
    nA = env.action_space.n

    # VI and PI
    pi_vi, V_vi, deltas_vi, t_vi = value_iteration(P, nS, nA, gamma=gamma, theta=theta)
    pi_pi, V_pi, hist_pi, t_pi = policy_iteration(P, nS, nA, gamma=gamma, theta=theta, in_place_eval=True)

    print(f"[{tag} (slippery={slippery})] VI: sweeps={len(deltas_vi)}, time={t_vi:.4f}s | "
          f"PI: iters={len(hist_pi)}, time={t_pi:.4f}s")

    # visualize 4x4 only (size mapping)
    size = 4 if map_name == "4x4" else int(map_name.split("x")[0])

    suffix = f"{tag}_{map_name}_{'slip' if slippery else 'noslip'}"
    plot_value_heatmap(V_vi, size, f"FrozenLake {map_name} (slippery={slippery}) — V* (VI)",
                       f"results/{suffix}_VI_V_heatmap.png")
    plot_policy_arrows(pi_vi, size, f"FrozenLake {map_name} (slippery={slippery}) — π* (VI)",
                       f"results/{suffix}_VI_policy.png")
    plot_value_heatmap(V_pi, size, f"FrozenLake {map_name} (slippery={slippery}) — V^π* (PI)",
                       f"results/{suffix}_PI_V_heatmap.png")
    plot_policy_arrows(pi_pi, size, f"FrozenLake {map_name} (slippery={slippery}) — π* (PI)",
                       f"results/{suffix}_PI_policy.png")
    plot_delta_curve(deltas_vi, f"FrozenLake {map_name} (slippery={slippery}) — VI Δ",
                     f"results/{suffix}_VI_delta.png")


def main():
    ensure_results()
    solve_frozenlake(map_name="4x4", slippery=True, tag="frozenlake")
    solve_frozenlake(map_name="4x4", slippery=False, tag="frozenlake")


if __name__ == "__main__":
    main()
