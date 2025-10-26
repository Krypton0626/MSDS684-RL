"""
Small runner that executes the Lab 1 experiment pipeline:
- runs epsilon-greedy experiments
- generates plots in lab1/figs

You can extend this later for UCB.
"""

from lab1.bandit_experiments import run_experiment, plot_results

if __name__ == "__main__":
    # While debugging, keep it light so it runs fast
    results = run_experiment(
        epsilons=(0.1, 0.2),  # test values
        n_runs=10,            # small for sanity test
        n_steps=200,          # small for sanity test
        n_arms=10,
    )

    plot_results(results, save_dir="lab1/figs")

    print("Runner finished. Check lab1/figs for output plots.")
