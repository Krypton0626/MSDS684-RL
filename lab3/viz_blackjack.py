import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from utils_mc import grid_value_maps, grid_policy_maps, smooth

def plot_surface(ps, ds, Z, title, out_path):
    # meshgrid: dealer on X, player on Y
    X, Y = np.meshgrid(ds, ps)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, linewidth=0, antialiased=True)
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_zlabel("Value (max_a Q)")
    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_policy(ps, ds, P, title, out_path):
    # P: 0=stick, 1=hit
    fig, ax = plt.subplots()
    im = ax.imshow(
        P,
        origin="lower",
        extent=[ds.min()-0.5, ds.max()+0.5, ps.min()-0.5, ps.max()+0.5],
        aspect="auto",
    )
    ax.set_xlabel("Dealer Showing")
    ax.set_ylabel("Player Sum")
    ax.set_title(f"{title} (0=stick, 1=hit)")
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_learning_curve(returns, out_path, smooth_k=5001):
    y = np.asarray(returns, dtype=float)
    x = np.arange(len(y))
    # pick a reasonable smoothing window relative to length
    k = min(smooth_k, max(1, (len(y)//10)*2 + 1))
    y_s = smooth(y, k=k)
    xs = np.arange(len(y_s))

    plt.figure()
    plt.plot(x, y, alpha=0.25, label="episode return")
    if len(y_s) > 1:
        plt.plot(xs, y_s, label=f"smoothed (k={k})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Learning Curve — Blackjack MC Control")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()

if __name__ == "__main__":
    Q = np.load("data/Q_blackjack.npy", allow_pickle=True).item()
    returns = np.load("data/episode_returns.npy")

    # Value surfaces
    ps, ds, Z_u, Z_n = grid_value_maps(Q)
    plot_surface(ps, ds, Z_u, "Value Surface (Usable Ace)", "figs/value_surface_usable_ace.png")
    plot_surface(ps, ds, Z_n, "Value Surface (No Usable Ace)", "figs/value_surface_no_ace.png")

    # Policy heatmaps
    ps, ds, P_u, P_n = grid_policy_maps(Q)
    plot_policy(ps, ds, P_u, "Policy (Usable Ace)", "figs/policy_usable_ace.png")
    plot_policy(ps, ds, P_n, "Policy (No Usable Ace)", "figs/policy_no_ace.png")

    # Learning curve
    plot_learning_curve(returns, "figs/learning_curve.png", smooth_k=5001)

    print("Saved plots to figs/")
