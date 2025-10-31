import numpy as np
import matplotlib.pyplot as plt


def plot_value_heatmap(V: np.ndarray, size: int, title: str, outpath: str):
    """
    Render V(s) as a size×size heatmap with numeric annotations.
    """
    grid = np.asarray(V, dtype=float).reshape(size, size)
    plt.figure()
    im = plt.imshow(grid, interpolation="nearest")
    plt.title(title)
    plt.colorbar(im, fraction=0.046, pad=0.04)

    # annotate each cell
    for (r, c), val in np.ndenumerate(grid):
        plt.text(c, r, f"{val:.1f}", ha="center", va="center")

    plt.xlabel("col")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_policy_arrows(policy: np.ndarray, size: int, title: str, outpath: str):
    """
    Draw greedy action arrows for each state on a size×size grid.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left.
    """
    # (dx, dy) for quiver uses x=columns, y=rows; invert y-axis to match matrix view
    arrow_vec = {
        0: (0, -1),  # Up
        1: (1,  0),  # Right
        2: (0,  1),  # Down
        3: (-1, 0),  # Left
    }

    X, Y, U, V = [], [], [], []
    for s in range(policy.shape[0]):
        r, c = divmod(s, size)
        a = int(np.argmax(policy[s]))
        dx, dy = arrow_vec[a]
        X.append(c)
        Y.append(r)
        U.append(dx)
        V.append(dy)

    plt.figure()
    plt.quiver(X, Y, U, V, angles="xy", scale_units="xy", scale=1)
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.xlim(-0.5, size - 0.5)
    plt.ylim(-0.5, size - 0.5)
    plt.xlabel("col")
    plt.ylabel("row")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()


def plot_delta_curve(deltas, title: str, outpath: str):
    """
    Plot max-norm value update Δ across sweeps/iterations.
    """
    plt.figure()
    plt.plot(deltas)
    plt.title(title)
    plt.xlabel("sweep")
    plt.ylabel("max Δ")
    plt.tight_layout()
    plt.savefig(outpath, dpi=150)
    plt.close()
