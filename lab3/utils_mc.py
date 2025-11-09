import numpy as np
from collections import defaultdict

def first_visit_indices(seq):
    """
    Return indices (as a set) for the first time each element appears in seq.
    Works for hashable items (e.g., tuples).
    """
    seen = set()
    out = set()
    for i, x in enumerate(seq):
        if x not in seen:
            seen.add(x)
            out.add(i)
    return out

def compute_returns(rewards, gamma=1.0):
    """Compute discounted returns G_t for a single episode."""
    G = 0.0
    returns = [0.0] * len(rewards)
    for t in range(len(rewards) - 1, -1, -1):
        G = rewards[t] + gamma * G
        returns[t] = G
    return returns

def smooth(y, k=101):
    """
    Moving-average smoothing. If k<=1 or k>len(y), returns y unchanged.
    """
    y = np.asarray(y, dtype=float)
    if k <= 1 or k > len(y):
        return y
    win = np.ones(int(k), dtype=float) / float(k)
    return np.convolve(y, win, mode="valid")

def init_q_default(num_actions):
    """Defaultdict for Q with zero-initialized action values."""
    return defaultdict(lambda: np.zeros(num_actions, dtype=float))

def epsilon_greedy_action(Q, state, epsilon, nA):
    """Sample an action ε-greedily from Q(state)."""
    if np.random.random() < epsilon:
        return np.random.randint(nA)
    return int(np.argmax(Q[state]))

def to_state_key(obs):
    """
    Blackjack-v1 observation is a tuple:
      (player_sum:int, dealer_showing:int, usable_ace:bool)
    Keep as-is so it's hashable for dict keys.
    """
    return obs

def value_from_q(Q):
    """V(s) = max_a Q(s,a) for states present in Q."""
    return {s: float(np.max(a_vals)) for s, a_vals in Q.items()}

def extract_policy(Q):
    """Greedy policy w.r.t. Q(s, a)."""
    return {s: int(np.argmax(a_vals)) for s, a_vals in Q.items()}

def grid_value_maps(Q):
    """
    Build value grids for visualization:
      player sum: 12..21 (rows), dealer: 1..10 (cols),
      separately for usable_ace True/False.
    """
    ps = np.arange(12, 22)
    ds = np.arange(1, 11)
    Z_usable = np.zeros((len(ps), len(ds)))
    Z_no = np.zeros((len(ps), len(ds)))
    for i, p in enumerate(ps):
        for j, d in enumerate(ds):
            for usable, Z in [(True, Z_usable), (False, Z_no)]:
                s = (int(p), int(d), bool(usable))
                if s in Q:
                    Z[i, j] = np.max(Q[s])
                else:
                    Z[i, j] = 0.0
    return ps, ds, Z_usable, Z_no

def grid_policy_maps(Q):
    """
    Build policy grids (0=stick, 1=hit) over same state lattice.
    """
    ps = np.arange(12, 22)
    ds = np.arange(1, 11)
    P_usable = np.zeros((len(ps), len(ds)), dtype=int)
    P_no = np.zeros((len(ps), len(ds)), dtype=int)
    for i, p in enumerate(ps):
        for j, d in enumerate(ds):
            for usable, P in [(True, P_usable), (False, P_no)]:
                s = (int(p), int(d), bool(usable))
                if s in Q:
                    P[i, j] = int(np.argmax(Q[s]))  # 0=stick, 1=hit
                else:
                    P[i, j] = 0
    return ps, ds, P_usable, P_no
