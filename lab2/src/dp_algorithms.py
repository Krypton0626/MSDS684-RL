from typing import Dict, List, Tuple
import numpy as np
import time

# Transition dictionary type:
# P[s][a] = list of (prob, next_state, reward, done)
TransitionDict = Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]


def policy_evaluation_sync(
    P: TransitionDict,
    nS: int,
    nA: int,
    policy: np.ndarray,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iter: int = 10000,
):
    """
    Synchronous (Jacobi) policy evaluation: uses previous-iteration values on RHS.
    Returns: (V, deltas, elapsed_seconds)
    """
    V = np.zeros(nS, dtype=float)
    deltas: List[float] = []
    t0 = time.time()

    for _ in range(max_iter):
        V_new = np.zeros_like(V)
        delta = 0.0
        for s in range(nS):
            v = 0.0
            for a in range(nA):
                pi = policy[s, a]
                if pi == 0.0:
                    continue
                for prob, ns, r, done in P[s][a]:
                    v += pi * prob * (r + gamma * (0.0 if done else V[ns]))
            V_new[s] = v
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        deltas.append(delta)
        if delta < theta:
            break

    t1 = time.time()
    return V, deltas, t1 - t0


def policy_evaluation_inplace(
    P: TransitionDict,
    nS: int,
    nA: int,
    policy: np.ndarray,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iter: int = 10000,
):
    """
    In-place (Gauss–Seidel) policy evaluation: updates V[s] immediately.
    Returns: (V, deltas, elapsed_seconds)
    """
    V = np.zeros(nS, dtype=float)
    deltas: List[float] = []
    t0 = time.time()

    for _ in range(max_iter):
        delta = 0.0
        for s in range(nS):
            old = V[s]
            v = 0.0
            for a in range(nA):
                pi = policy[s, a]
                if pi == 0.0:
                    continue
                for prob, ns, r, done in P[s][a]:
                    v += pi * prob * (r + gamma * (0.0 if done else V[ns]))
            V[s] = v
            delta = max(delta, abs(old - v))
        deltas.append(delta)
        if delta < theta:
            break

    t1 = time.time()
    return V, deltas, t1 - t0


def policy_improvement(
    P: TransitionDict,
    nS: int,
    nA: int,
    V: np.ndarray,
    gamma: float = 0.9,
):
    """
    Greedy improvement: π'(s) = argmax_a Q^π(s,a).
    Returns a deterministic one-hot policy of shape (nS, nA).
    """
    policy = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        q = np.zeros(nA, dtype=float)
        for a in range(nA):
            for prob, ns, r, done in P[s][a]:
                q[a] += prob * (r + gamma * (0.0 if done else V[ns]))
        best = int(np.argmax(q))
        policy[s, best] = 1.0
    return policy


def policy_iteration(
    P: TransitionDict,
    nS: int,
    nA: int,
    gamma: float = 0.9,
    theta: float = 1e-6,
    in_place_eval: bool = True,
    max_eval_iter: int = 10000,
):
    """
    Classic Policy Iteration:
      - Evaluate current policy (sync or in-place)
      - Greedy improve
      - Repeat until stable
    Returns: (policy, V, history, total_eval_time)
      history: list of tuples (iter_idx, sweeps_used, last_delta, eval_time_sec)
    """
    policy = np.ones((nS, nA), dtype=float) / nA  # start uniform
    total_eval_time = 0.0
    history = []

    while True:
        if in_place_eval:
            V, deltas, t_eval = policy_evaluation_inplace(P, nS, nA, policy, gamma, theta, max_eval_iter)
        else:
            V, deltas, t_eval = policy_evaluation_sync(P, nS, nA, policy, gamma, theta, max_eval_iter)

        total_eval_time += t_eval
        old_greedy = np.argmax(policy, axis=1)
        policy = policy_improvement(P, nS, nA, V, gamma)
        new_greedy = np.argmax(policy, axis=1)
        changed = not np.array_equal(old_greedy, new_greedy)

        history.append((len(history) + 1, len(deltas), (deltas[-1] if deltas else None), t_eval))
        if not changed:
            break

    return policy, V, history, total_eval_time


def value_iteration(
    P: TransitionDict,
    nS: int,
    nA: int,
    gamma: float = 0.9,
    theta: float = 1e-6,
    max_iter: int = 10000,
):
    """
    Value Iteration with Δ tracking.
    Returns: (greedy_policy, V, deltas, elapsed_seconds)
    """
    V = np.zeros(nS, dtype=float)
    deltas: List[float] = []
    t0 = time.time()

    for _ in range(max_iter):
        delta = 0.0
        for s in range(nS):
            v_old = V[s]
            q = np.zeros(nA, dtype=float)
            for a in range(nA):
                for prob, ns, r, done in P[s][a]:
                    q[a] += prob * (r + gamma * (0.0 if done else V[ns]))
            V[s] = np.max(q)
            delta = max(delta, abs(v_old - V[s]))
        deltas.append(delta)
        if delta < theta:
            break

    t1 = time.time()
    elapsed = t1 - t0

    # Extract greedy policy from V
    policy = np.zeros((nS, nA), dtype=float)
    for s in range(nS):
        q = np.zeros(nA, dtype=float)
        for a in range(nA):
            for prob, ns, r, done in P[s][a]:
                q[a] += prob * (r + gamma * (0.0 if done else V[ns]))
        policy[s, int(np.argmax(q))] = 1.0

    return policy, V, deltas, elapsed
