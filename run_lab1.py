from lab1.bandit_experiments import run_experiment, plot_results

if __name__ == "__main__":
    # Final graded config:
    results = run_experiment(
        egreedy_epsilons=(0.01, 0.1, 0.2),
        ucb_cs=(1.0, 2.0),
        n_runs=1000,
        n_steps=2000,
        n_arms=10,
    )

    plot_results(results, save_dir="lab1/figs")
    print("Done. Final comparison plots saved in lab1/figs/")
