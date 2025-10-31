# run_lab2.py — convenience runner from repo root
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.run_gridworld import main as run_gridworld_main
from experiments.run_frozenlake import main as run_frozenlake_main

if __name__ == "__main__":
    print("== Running GridWorld experiments ==")
    run_gridworld_main()
    print("== Running FrozenLake experiments ==")
    run_frozenlake_main()
    print("== All Lab 2 runs complete. Results saved in ./results ==")
