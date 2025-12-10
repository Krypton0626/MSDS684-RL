"""
run_lab7.py (root-level)

Entry point for running all Lab 7 experiments, including:
- Taxi-v3 Dyna-Q, Dyna-Q+, Prioritized Sweeping
- Optional MountainCar-v0 neural dynamics + model-based policy gradient (if PyTorch is available)
"""

from lab7.runner_lab7 import run_all_lab7_experiments


if __name__ == "__main__":
    run_all_lab7_experiments()
