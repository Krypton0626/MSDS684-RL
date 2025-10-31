import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass
class GridSpec:
    size: int = 4
    terminals: Tuple[int, ...] = (0, 15)
    obstacles: Tuple[int, ...] = ()
    step_reward: float = -1.0
    goal_reward: float = 0.0
    intended_prob: float = 1.0  # 1.0 = deterministic, <1.0 = stochastic
    stochastic: bool = False


class GridWorld:
    """
    GridWorld environment with deterministic or stochastic transitions.
    States numbered row-major from 0 to size*size - 1.
    Actions: 0=Up, 1=Right, 2=Down, 3=Left.
    """

    def __init__(self, spec: GridSpec):
        self.spec = spec
        self.size = spec.size
        self.nS = self.size * self.size
        self.nA = 4
        self.terminals = spec.terminals
        self.obstacles = spec.obstacles
        self.P = self._build_transitions()

    def _build_transitions(self) -> Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]]:
        """
        Build transition dynamics: P[s][a] = list of (prob, next_state, reward, done).
        Deterministic or stochastic depending on spec.
        """
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}

            # terminal or obstacle: self-loop
            if s in self.terminals or s in self.obstacles:
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue

            r, c = divmod(s, self.size)
            for a, (dr, dc) in enumerate(deltas):
                nr, nc = max(0, min(self.size - 1, r + dr)), max(0, min(self.size - 1, c + dc))
                ns = nr * self.size + nc

                # handle obstacles → bounce back
                if ns in self.obstacles:
                    ns = s

                reward = self.spec.goal_reward if ns in self.terminals else self.spec.step_reward
                done = ns in self.terminals

                # deterministic transitions
                if not self.spec.stochastic:
                    P[s][a] = [(1.0, ns, reward, done)]

                else:
                    # stochastic: intended p=intended_prob, slip left/right equally
                    intended_prob = float(self.spec.intended_prob)
                    right_perp = deltas[(a + 1) % 4]
                    left_perp  = deltas[(a - 1) % 4]
                    moves = [(dr, dc), right_perp, left_perp]

                    leftover = max(0.0, 1.0 - intended_prob)
                    p_slip = leftover / (len(moves) - 1)
                    probs = [intended_prob, p_slip, p_slip]

                    # accumulate transitions if same cell hit twice
                    bucket = {}
                    for p, (mdr, mdc) in zip(probs, moves):
                        nr2, nc2 = max(0, min(self.size - 1, r + mdr)), max(0, min(self.size - 1, c + mdc))
                        ns2 = nr2 * self.size + nc2
                        if ns2 in self.obstacles:
                            ns2 = s
                        rew2 = self.spec.goal_reward if ns2 in self.terminals else self.spec.step_reward
                        done2 = ns2 in self.terminals
                        key = (ns2, rew2, done2)
                        bucket[key] = bucket.get(key, 0.0) + p

                    P[s][a] = [(p, ns2, rew2, done2) for (ns2, rew2, done2), p in bucket.items()]

        return P

    def display_transition_summary(self):
        """Quick summary of transitions for debugging."""
        for s in range(self.nS):
            print(f"State {s}:")
            for a in range(self.nA):
                print(f"  Action {a}: {self.P[s][a]}")
            print()
