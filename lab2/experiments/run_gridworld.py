import os
from src.gridworld import GridWorld, GridSpec
from src.dp_algorithms import policy_iteration, value_iteration
from src.visualize import plot_value_heatmap, plot_policy_arrows, plot_delta_curve


def ensure_results():
    os.makedirs("results", exist_ok=True)


def run_case(spec: GridSpec, tag: str, gamma=0.9, theta=1e-6):
    env = GridWorld(spec)
    P, nS, nA = env.P, env.nS, env.nA

    # Policy Iteration (in-place eval by default)
    pi_pol, V_pol, hist_pol, t_pol = policy_iteration(
        P, nS, nA, gamma=gamma, theta=theta, in_place_eval=True
    )

    # Value Iteration
    pi_val, V_val, deltas_val, t_val = value_iteration(
        P, nS, nA, gamma=gamma, theta=theta
    )

    print(f"[{tag}] Policy Iteration: iters={len(hist_pol)}, "
          f"sweeps={[h[1] for h in hist_pol]}, time={t_pol:.4f}s")
    print(f"[{tag}] Value Iteration : sweeps={len(deltas_val)}, time={t_val:.4f}s")

    # Plots
    plot_value_heatmap(V_pol, env.size, f"{tag} — V^π* (Policy Iteration)",
                       f"results/{tag}_PI_V_heatmap.png")
    plot_policy_arrows(pi_pol, env.size, f"{tag} — π* (Policy Iteration)",
                       f"results/{tag}_PI_policy.png")

    plot_value_heatmap(V_val, env.size, f"{tag} — V* (Value Iteration)",
                       f"results/{tag}_VI_V_heatmap.png")
    plot_policy_arrows(pi_val, env.size, f"{tag} — π* (Value Iteration)",
                       f"results/{tag}_VI_policy.png")
    plot_delta_curve(deltas_val, f"{tag} — VI Δ per sweep",
                     f"results/{tag}_VI_delta.png")


def main():
    ensure_results()

    # Deterministic 5x5
    spec_det = GridSpec(
        size=5, terminals=(0, 24), obstacles=(), step_reward=-1.0,
        goal_reward=0.0, intended_prob=1.0, stochastic=False
    )
    run_case(spec_det, tag="grid_det_5x5", gamma=0.95, theta=1e-6)

    # Stochastic 5x5 (70% intended, 15% perpendicular each)
    spec_sto = GridSpec(
        size=5, terminals=(0, 24), obstacles=(), step_reward=-1.0,
        goal_reward=0.0, intended_prob=0.7, stochastic=True
    )
    run_case(spec_sto, tag="grid_sto_5x5", gamma=0.95, theta=1e-6)


if __name__ == "__main__":
    main()
