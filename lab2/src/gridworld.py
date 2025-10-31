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
    intended_prob: float = 1.0  # 1.0 = deterministic
    stochastic: bool = False


class GridWorld:
    """
    Simple GridWorld with deterministic or stochastic transitions.
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
        P: Dict[int, Dict[int, List[Tuple[float, int, float, bool]]]] = {}
        deltas = [(-1, 0), (0, 1), (1, 0), (0, -1)]  # Up, Right, Down, Left

        for s in range(self.nS):
            P[s] = {a: [] for a in range(self.nA)}

            if s in self.terminals or s in self.obstacles:
                for a in range(self.nA):
                    P[s][a] = [(1.0, s, 0.0, True)]
                continue

            r, c = divmod(s, self.size)
            for a, (dr, dc) in enumerate(deltas):
                nr, nc = max(0, min(self.size - 1, r + dr)), max(0, min(self.size - 1, c + dc))
                ns = nr * self.size + nc
                reward = self.spec.goal_reward if ns in self.terminals else self.spec.step_reward
                done = ns in self.terminals

                # deterministic
                if not self.spec.stochastic:
                    P[s][a] = [(1.0, ns, reward, done)]
                else:
                    # stochastic variant will be added in next commit
                    pass
        return P

    def display_transition_summary(self):
        """Quick text summary of transitions."""
        for s in range(self.nS):
            print(f"State {s}:")
            for a in range(self.nA):
                print(f"  Action {a}: {self.P[s][a]}")
            print()
