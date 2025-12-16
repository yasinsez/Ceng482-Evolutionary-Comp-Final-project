"""
Experiment runners for PSO-SVM feature selection.

These helpers connect:
- Dataset loading,
- PSO-based feature selection, and
- SVM fitness evaluation,
and provide utilities to repeat the algorithm multiple times (e.g., 30
runs as in your project) and average the results.
"""

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from .config import PSOConfig, SVMConfig, get_default_pso_config, get_default_svm_config
from .data_loading import get_dataset
from .pso import run_pso_feature_selection
from .svm_fitness import evaluate_feature_subset_with_svm


def build_fitness_function(
    X: np.ndarray,
    y: np.ndarray,
    svm_config: Optional[SVMConfig] = None,
) -> Callable[[np.ndarray], float]:
    """
    Create a fitness function that maps a feature mask to an SVM accuracy.

    Parameters
    ----------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Class labels.
    svm_config : SVMConfig, optional
        SVM hyperparameters. If None, use `get_default_svm_config()`.

    Returns
    -------
    fitness_fn : callable
        Function: feature_mask -> fitness (float).

    Notes
    -----
    This is a thin wrapper that closes over X, y, and SVM configuration,
    and then calls `evaluate_feature_subset_with_svm`.
    """
    if svm_config is None:
        svm_config = get_default_svm_config()

    def fitness_fn(feature_mask: np.ndarray) -> float:
        return evaluate_feature_subset_with_svm(
            X=X,
            y=y,
            feature_mask=feature_mask,
            svm_config=svm_config,
        )

    return fitness_fn


def _make_timestamp_str() -> str:
    """Return a compact timestamp string suitable for filenames."""
    from datetime import datetime as _dt

    return _dt.now().strftime("%Y%m%d_%H%M%S")


def _get_results_base_dir(dataset_name: str, results_dir: Optional[str]) -> str:
    """
    Compute a directory path where results for a given dataset will be stored.

    The directory is created if it does not already exist.
    """
    from pathlib import Path as _Path

    if results_dir is not None:
        base = _Path(results_dir)
    else:
        # Default: place results under "Final Project/results/<dataset_name>"
        base = _Path(__file__).resolve().parent.parent / "results" / dataset_name

    base.mkdir(parents=True, exist_ok=True)
    return str(base)


def _save_single_run_result(
    output_dir: str,
    dataset_name: str,
    run_index: int,
    run_results: Dict[str, Any],
    pso_config: PSOConfig,
    svm_config: SVMConfig,
    base_random_seed: Optional[int],
    random_seed: Optional[int],
) -> str:
    """
    Save the result of a single run to a JSON file under `output_dir`.

    The file includes:
    - dataset name and run index,
    - timestamp and seeds,
    - PSO and SVM hyperparameters,
    - best fitness, number of selected features,
    - best feature mask and fitness history (if available).
    """
    import json as _json
    from dataclasses import asdict as _asdict
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    out_path = _Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = _make_timestamp_str()
    file_name = f"{dataset_name}_run{run_index:03d}_{timestamp}.json"
    full_path = out_path / file_name

    record: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "run_index": run_index,
        "timestamp": _dt.now().isoformat(),
        "base_random_seed": base_random_seed,
        "random_seed": random_seed,
        "pso_config": _asdict(pso_config),
        "svm_config": _asdict(svm_config),
    }

    if "best_fitness" in run_results:
        record["best_fitness"] = float(run_results["best_fitness"])
    if "num_selected_features" in run_results:
        record["num_selected_features"] = int(run_results["num_selected_features"])

    best_position = run_results.get("best_position")
    if best_position is not None:
        record["best_position"] = np.asarray(best_position, dtype=int).tolist()

    fitness_history = run_results.get("fitness_history")
    if fitness_history is not None:
        record["fitness_history"] = np.asarray(fitness_history, dtype=float).tolist()

    with full_path.open("w", encoding="utf-8") as f:
        _json.dump(record, f, indent=2)

    return str(full_path)


def _save_experiment_summary(
    output_dir: str,
    dataset_name: str,
    num_runs: int,
    summary: Dict[str, Any],
) -> str:
    """
    Save the aggregated summary of multiple runs as a JSON file.
    """
    import json as _json
    from datetime import datetime as _dt
    from pathlib import Path as _Path

    out_path = _Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    timestamp = _make_timestamp_str()
    file_name = f"summary_{dataset_name}_{num_runs}runs_{timestamp}.json"
    full_path = out_path / file_name

    summary_with_meta: Dict[str, Any] = dict(summary)
    summary_with_meta.setdefault("timestamp", _dt.now().isoformat())

    with full_path.open("w", encoding="utf-8") as f:
        _json.dump(summary_with_meta, f, indent=2)

    return str(full_path)


def run_single_experiment(
    dataset_name: str,
    pso_config: Optional[PSOConfig] = None,
    svm_config: Optional[SVMConfig] = None,
    random_state: Optional[int] = None,
) -> Dict[str, np.ndarray | float]:
    """
    Run a single PSO-SVM feature selection experiment on one dataset.

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load (e.g., "wdbc", "wine", "sonar").
    pso_config : PSOConfig, optional
        PSO hyperparameters. If None, use `get_default_pso_config()`.
    svm_config : SVMConfig, optional
        SVM hyperparameters. If None, use `get_default_svm_config()`.
    random_state : int, optional
        Seed for reproducibility.

    Returns
    -------
    results : dict
        Expected keys:
        - "best_position": np.ndarray, best feature mask.
        - "best_fitness": float, fitness (e.g., accuracy) of best mask.
        - "num_selected_features": int, number of features selected.
        - "fitness_history": np.ndarray, per-iteration fitness (optional).

    Notes
    -----
    This is a stub that wires together other components; you still need
    to implement the PSO and SVM functions for it to run end-to-end.
    """
    if pso_config is None:
        pso_config = get_default_pso_config()

    X, y = get_dataset(dataset_name)
    num_features = X.shape[1]

    fitness_fn = build_fitness_function(X, y, svm_config=svm_config)

    pso_results = run_pso_feature_selection(
        num_features=num_features,
        fitness_fn=fitness_fn,
        pso_config=pso_config,
        random_state=random_state,
    )

    best_position = pso_results.get("best_position")
    best_fitness = float(pso_results.get("best_fitness", np.nan))
    num_selected_features = int(best_position.sum()) if best_position is not None else 0

    return {
        "best_position": best_position,
        "best_fitness": best_fitness,
        "num_selected_features": num_selected_features,
        "fitness_history": pso_results.get("fitness_history"),
    }


def run_multiple_experiments(
    dataset_name: str,
    num_runs: int = 30,
    pso_config: Optional[PSOConfig] = None,
    svm_config: Optional[SVMConfig] = None,
    base_random_seed: Optional[int] = 42,
    save_results: bool = True,
    results_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Run the PSO-SVM feature selection algorithm multiple times and
    compute average performance.

    Parameters
    ----------
    dataset_name : str
        Dataset to use.
    num_runs : int, default 30
        Number of independent PSO runs.
    pso_config : PSOConfig, optional
        PSO hyperparameters.
    svm_config : SVMConfig, optional
        SVM hyperparameters.
    base_random_seed : int, optional
        Base seed; each run can offset it for reproducibility.
    save_results : bool, default True
        If True, save every individual run and the aggregated summary
        to disk so that no results are lost.
    results_dir : str, optional
        Base directory where results will be stored. If None, a default
        `results/` directory under `Final Project` is used.

    Returns
    -------
    summary : dict
        Keys (suggested):
        - "dataset_name": str
        - "num_runs": int
        - "average_best_fitness": float
        - "std_best_fitness": float
        - "average_num_selected_features": float
        - "all_best_fitness": List[float]
        - "all_num_selected_features": List[float]
        - "results_dir": str or None
        - "run_result_files": List[str]
        - "summary_file": str or None

    Notes
    -----
    This is a stub: it assumes `run_single_experiment` is correctly
    implemented. You will likely want to extend the returned summary
    with more statistics (min, max, etc.) similar to the analysis in
    the PSO-SVM paper.
    """
    if num_runs <= 0:
        raise ValueError("num_runs must be positive.")

    # Ensure we have concrete configs so we can also log them.
    if pso_config is None:
        pso_config = get_default_pso_config()
    if svm_config is None:
        svm_config = get_default_svm_config()

    # Prepare output directory if saving is enabled.
    results_base_dir: Optional[str] = None
    if save_results:
        results_base_dir = _get_results_base_dir(
            dataset_name=dataset_name, results_dir=results_dir
        )

    best_fitness_values: List[float] = []
    num_features_values: List[int] = []
    run_result_files: List[str] = []

    for run_idx in range(num_runs):
        seed = None
        if base_random_seed is not None:
            seed = base_random_seed + run_idx

        run_results = run_single_experiment(
            dataset_name=dataset_name,
            pso_config=pso_config,
            svm_config=svm_config,
            random_state=seed,
        )

        best_fitness_values.append(float(run_results["best_fitness"]))
        num_features_values.append(int(run_results["num_selected_features"]))

        if save_results and results_base_dir is not None:
            run_path = _save_single_run_result(
                output_dir=results_base_dir,
                dataset_name=dataset_name,
                run_index=run_idx,
                run_results=run_results,
                pso_config=pso_config,
                svm_config=svm_config,
                base_random_seed=base_random_seed,
                random_seed=seed,
            )
            run_result_files.append(run_path)

    average_best_fitness = float(np.mean(best_fitness_values))
    std_best_fitness = float(np.std(best_fitness_values))
    average_num_selected_features = float(np.mean(num_features_values))

    summary: Dict[str, Any] = {
        "dataset_name": dataset_name,
        "num_runs": num_runs,
        "average_best_fitness": average_best_fitness,
        "std_best_fitness": std_best_fitness,
        "average_num_selected_features": average_num_selected_features,
        "all_best_fitness": best_fitness_values,
        "all_num_selected_features": num_features_values,
    }

    summary_file: Optional[str] = None
    if save_results and results_base_dir is not None:
        summary_file = _save_experiment_summary(
            output_dir=results_base_dir,
            dataset_name=dataset_name,
            num_runs=num_runs,
            summary=summary,
        )

    summary["results_dir"] = results_base_dir
    summary["run_result_files"] = run_result_files
    summary["summary_file"] = summary_file

    return summary

