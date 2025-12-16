"""
Entry point script for running PSO-SVM feature selection experiments.

This script is intentionally minimal and relies on stub implementations in
the `pso_svm_fs` package. Once you implement the underlying functions, you can
use this as a starting point to reproduce experiments similar to those in:
`/papers/Feature%20Selection%20using%20PSO-SVM.pdf`.
"""

from __future__ import annotations

from pso_svm_fs.experiment import run_multiple_experiments


def main() -> None:
    """
    Run PSO-SVM feature selection on a chosen dataset multiple times
    and print a summary of the results.

    Notes
    -----
    This is a stub-style main function. Adjust dataset name, number of runs,
    and configurations as needed for your final experiments.
    """
    dataset_name = "wdbc"  # e.g., "wdbc", "wine", "sonar"
    num_runs = 30

    summary = run_multiple_experiments(
        dataset_name=dataset_name,
        num_runs=num_runs,
        save_results=True,
    )

    print("PSO-SVM Feature Selection Summary")
    print("---------------------------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    # Highlight where detailed results have been stored on disk.
    summary_file = summary.get("summary_file")
    if summary_file:
        print(f"\nDetailed summary saved to: {summary_file}")


if __name__ == "__main__":
    main()


