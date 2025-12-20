"""
Analyze PSO-SVM feature selection results and create comparison figures.

Outputs (per dataset):
- Accuracy comparison (PSO-selected vs all-features baseline) + feature-count summary
- Wilcoxon signed-rank test (paired, PSO-selected vs baseline)
- Convergence graph (average best fitness over iterations from saved fitness_history)

Important note about accuracy:
The JSON field `best_fitness` is the fitness value observed during the PSO run.
Because PSO evaluates many masks, that value can be noisy if CV splits vary.
For fair comparisons and for Wilcoxon testing, this script recomputes:
  - accuracy_selected: SVM accuracy using the saved best_position mask
  - accuracy_baseline: SVM accuracy using ALL features
using the same deterministic CV setup with random_state = random_seed of the run.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Ensure repo root is on sys.path so `pso_svm_fs` can be imported when running from `scripts/`.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Make Matplotlib cache writable (helps in restricted/sandboxed environments).
os.environ.setdefault("MPLCONFIGDIR", str(_REPO_ROOT / ".matplotlib_cache"))

from pso_svm_fs.config import SVMConfig, get_svm_config_for_dataset
from pso_svm_fs.data_loading import get_dataset
from pso_svm_fs.svm_fitness import evaluate_feature_subset_with_svm


def _load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _find_run_files(results_root: Path, dataset: str) -> List[Path]:
    ds_dir = results_root / dataset
    if not ds_dir.exists():
        raise FileNotFoundError(f"Dataset results directory not found: {ds_dir}")

    candidates = list(ds_dir.glob(f"{dataset}_run*.json"))
    if not candidates:
        raise FileNotFoundError(f"No run JSON files found under: {ds_dir}")

    # Deduplicate by run_index (some folders include multiple run000 files).
    # Keep the newest file by modification time.
    best_by_index: Dict[int, Path] = {}
    for p in candidates:
        try:
            rec = _load_json(p)
            idx = int(rec.get("run_index"))
        except Exception:
            continue

        prev = best_by_index.get(idx)
        if prev is None:
            best_by_index[idx] = p
        else:
            if p.stat().st_mtime > prev.stat().st_mtime:
                best_by_index[idx] = p

    run_files = [best_by_index[k] for k in sorted(best_by_index.keys())]
    if not run_files:
        raise FileNotFoundError(f"Run files exist but none could be parsed under: {ds_dir}")

    return run_files


def _build_svm_config_from_run(run: Dict[str, Any], dataset_name: str) -> SVMConfig:
    cfg_dict = run.get("svm_config")
    if isinstance(cfg_dict, dict):
        # Defensive: only pass known keys
        allowed = {"C", "kernel", "gamma", "r"}
        clean = {k: cfg_dict.get(k) for k in allowed if k in cfg_dict}
        return SVMConfig(**clean)
    return get_svm_config_for_dataset(dataset_name)


def _recompute_accuracies(
    X: np.ndarray,
    y: np.ndarray,
    best_position: np.ndarray,
    svm_config: SVMConfig,
    num_folds: int,
    random_seed: int | None,
    n_jobs: int,
) -> Tuple[float, float]:
    """Return (accuracy_selected, accuracy_baseline_all_features)."""
    best_mask = np.asarray(best_position, dtype=int).ravel()
    all_mask = np.ones(X.shape[1], dtype=int)

    acc_selected = evaluate_feature_subset_with_svm(
        X=X,
        y=y,
        feature_mask=best_mask,
        svm_config=svm_config,
        num_folds=num_folds,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    acc_baseline = evaluate_feature_subset_with_svm(
        X=X,
        y=y,
        feature_mask=all_mask,
        svm_config=svm_config,
        num_folds=num_folds,
        random_state=random_seed,
        n_jobs=n_jobs,
    )
    return float(acc_selected), float(acc_baseline)


def _wilcoxon_signed_rank(x: np.ndarray, y: np.ndarray) -> Dict[str, Any]:
    """
    Paired Wilcoxon signed-rank test.

    We try SciPy first (more robust); if not available, raise a clear error.
    """
    try:
        from scipy.stats import wilcoxon  # type: ignore
    except Exception as e:  # pragma: no cover
        raise RuntimeError(
            "SciPy is required for Wilcoxon test. Install it with: pip install scipy"
        ) from e

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    if x.shape != y.shape:
        raise ValueError("Wilcoxon test requires paired samples with same shape.")

    # Remove exact ties (differences = 0) to avoid SciPy warnings and to match common practice
    d = x - y
    keep = d != 0.0
    x2 = x[keep]
    y2 = y[keep]

    if x2.size < 5:
        return {
            "n_total": int(x.size),
            "n_used": int(x2.size),
            "warning": "Too few non-tied pairs for a reliable Wilcoxon test.",
            "statistic": None,
            "p_value": None,
        }

    res = wilcoxon(x2, y2, alternative="two-sided", zero_method="wilcox")
    return {
        "n_total": int(x.size),
        "n_used": int(x2.size),
        "statistic": float(res.statistic),
        "p_value": float(res.pvalue),
    }


def _ensure_outdir(outdir: Path) -> None:
    outdir.mkdir(parents=True, exist_ok=True)


def _plot_comparison(
    *,
    dataset: str,
    outdir: Path,
    selected_acc: np.ndarray,
    baseline_acc: np.ndarray,
    feature_counts: np.ndarray,
    total_features: int,
) -> Path:
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(10, 7))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0], width_ratios=[1.2, 1.0])

    ax_box = fig.add_subplot(gs[0, 0])
    ax_scatter = fig.add_subplot(gs[0, 1])
    ax_feat = fig.add_subplot(gs[1, :])

    # Accuracy boxplot
    ax_box.boxplot([selected_acc, baseline_acc], tick_labels=["PSO-selected", "All features"])
    ax_box.set_title(f"{dataset.upper()}: Accuracy (10-fold CV, re-evaluated)")
    ax_box.set_ylabel("Accuracy")
    ax_box.grid(True, axis="y", alpha=0.25)

    # Scatter: accuracy vs number of features
    ax_scatter.scatter(feature_counts, selected_acc, alpha=0.8)
    ax_scatter.axhline(float(np.mean(baseline_acc)), linestyle="--", linewidth=1.5, label="Baseline mean")
    ax_scatter.set_title("Accuracy vs selected feature count")
    ax_scatter.set_xlabel("# selected features")
    ax_scatter.set_ylabel("Accuracy")
    ax_scatter.grid(True, alpha=0.25)
    ax_scatter.legend(loc="lower right")

    # Feature count plot
    ax_feat.boxplot([feature_counts], tick_labels=["PSO-selected"])
    ax_feat.axhline(total_features, color="red", linestyle="--", linewidth=1.5, label=f"Total features = {total_features}")
    ax_feat.set_title("Selected feature count distribution (30 runs)")
    ax_feat.set_ylabel("# features")
    ax_feat.grid(True, axis="y", alpha=0.25)
    ax_feat.legend(loc="upper right")

    fig.tight_layout()
    out_path = outdir / f"{dataset}_comparison_accuracy_features.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def _plot_convergence(
    *,
    dataset: str,
    outdir: Path,
    histories: List[np.ndarray],
) -> Path | None:
    import matplotlib.pyplot as plt

    if not histories:
        return None

    # Align by minimum length
    min_len = min(h.size for h in histories)
    if min_len <= 1:
        return None

    H = np.vstack([h[:min_len] for h in histories]).astype(float)
    mean = np.mean(H, axis=0)
    std = np.std(H, axis=0)
    iters = np.arange(min_len)

    fig, ax = plt.subplots(figsize=(10, 4.5))
    ax.plot(iters, mean, label="Mean best fitness")
    ax.fill_between(iters, mean - std, mean + std, alpha=0.2, label="±1 std")
    ax.set_title(f"{dataset.upper()}: Convergence (best fitness history)")
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Best fitness (reported during PSO)")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right")
    fig.tight_layout()

    out_path = outdir / f"{dataset}_convergence.png"
    fig.savefig(out_path, dpi=200)
    plt.close(fig)
    return out_path


def analyze_dataset(
    *,
    results_root: Path,
    dataset: str,
    outdir: Path,
    num_folds: int,
    n_jobs: int,
) -> Dict[str, Any]:
    run_files = _find_run_files(results_root, dataset)
    X, y = get_dataset(dataset)
    total_features = int(X.shape[1])

    selected_acc: List[float] = []
    baseline_acc: List[float] = []
    reported_best_fitness: List[float] = []
    selected_features: List[int] = []
    histories: List[np.ndarray] = []
    used_run_files: List[str] = []

    for rf in run_files:
        run = _load_json(rf)
        best_position = np.asarray(run.get("best_position"), dtype=int)
        random_seed = run.get("random_seed")
        svm_cfg = _build_svm_config_from_run(run, dataset)

        acc_sel, acc_base = _recompute_accuracies(
            X=X,
            y=y,
            best_position=best_position,
            svm_config=svm_cfg,
            num_folds=num_folds,
            random_seed=int(random_seed) if random_seed is not None else None,
            n_jobs=n_jobs,
        )
        selected_acc.append(acc_sel)
        baseline_acc.append(acc_base)
        reported_best_fitness.append(float(run.get("best_fitness", np.nan)))
        selected_features.append(int(run.get("num_selected_features", int(best_position.sum()))))

        fh = run.get("fitness_history")
        if fh is not None:
            histories.append(np.asarray(fh, dtype=float))

        used_run_files.append(str(rf))

    selected_acc_arr = np.asarray(selected_acc, dtype=float)
    baseline_acc_arr = np.asarray(baseline_acc, dtype=float)
    feature_counts_arr = np.asarray(selected_features, dtype=int)

    wilcoxon = _wilcoxon_signed_rank(selected_acc_arr, baseline_acc_arr)

    _ensure_outdir(outdir)
    comparison_fig = _plot_comparison(
        dataset=dataset,
        outdir=outdir,
        selected_acc=selected_acc_arr,
        baseline_acc=baseline_acc_arr,
        feature_counts=feature_counts_arr,
        total_features=total_features,
    )
    convergence_fig = _plot_convergence(dataset=dataset, outdir=outdir, histories=histories)

    summary: Dict[str, Any] = {
        "dataset": dataset,
        "num_runs": int(len(selected_acc_arr)),
        "num_folds": int(num_folds),
        "total_features": total_features,
        "svm_config_default_for_dataset": asdict(get_svm_config_for_dataset(dataset)),
        "selected_accuracy": {
            "mean": float(np.mean(selected_acc_arr)),
            "std": float(np.std(selected_acc_arr)),
            "min": float(np.min(selected_acc_arr)),
            "max": float(np.max(selected_acc_arr)),
        },
        "baseline_all_features_accuracy": {
            "mean": float(np.mean(baseline_acc_arr)),
            "std": float(np.std(baseline_acc_arr)),
            "min": float(np.min(baseline_acc_arr)),
            "max": float(np.max(baseline_acc_arr)),
        },
        "selected_feature_count": {
            "mean": float(np.mean(feature_counts_arr)),
            "std": float(np.std(feature_counts_arr)),
            "min": int(np.min(feature_counts_arr)),
            "max": int(np.max(feature_counts_arr)),
        },
        "wilcoxon_signed_rank_selected_vs_baseline": wilcoxon,
        "files": {
            "comparison_figure": str(comparison_fig),
            "convergence_figure": str(convergence_fig) if convergence_fig else None,
        },
        "per_run": {
            "run_files": used_run_files,
            "reported_best_fitness": [float(x) for x in reported_best_fitness],
            "reevaluated_selected_accuracy": [float(x) for x in selected_acc_arr.tolist()],
            "reevaluated_baseline_accuracy": [float(x) for x in baseline_acc_arr.tolist()],
            "selected_feature_count": [int(x) for x in feature_counts_arr.tolist()],
        },
    }

    out_json = outdir / f"{dataset}_analysis_summary.json"
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PSO-SVM results and create figures.")
    parser.add_argument(
        "--results-root",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "results"),
        help="Path to the results directory (default: <repo>/results).",
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default=str(Path(__file__).resolve().parent.parent / "analysis_outputs"),
        help="Output directory for figures and summary JSONs.",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["wdbc", "wine", "sonar"],
        help="Datasets to analyze (default: wdbc wine sonar).",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=10,
        help="Number of CV folds for re-evaluating accuracies (default: 10).",
    )
    parser.add_argument(
        "--n-jobs",
        type=int,
        default=1,
        help="Parallel jobs for cross-validation. Use 1 if you hit joblib/sandbox limits (default: 1).",
    )

    args = parser.parse_args()
    results_root = Path(args.results_root).expanduser().resolve()
    outdir = Path(args.outdir).expanduser().resolve()
    datasets = [d.lower().strip() for d in args.datasets]

    _ensure_outdir(outdir)

    all_summaries: Dict[str, Any] = {}
    for ds in datasets:
        ds_out = outdir / ds
        _ensure_outdir(ds_out)
        summary = analyze_dataset(
            results_root=results_root,
            dataset=ds,
            outdir=ds_out,
            num_folds=int(args.num_folds),
            n_jobs=int(args.n_jobs),
        )
        all_summaries[ds] = summary

        # Print a short console summary
        w = summary["wilcoxon_signed_rank_selected_vs_baseline"]
        print(f"\n=== {ds.upper()} ===")
        print(f"Selected acc mean±std: {summary['selected_accuracy']['mean']:.4f} ± {summary['selected_accuracy']['std']:.4f}")
        print(f"Baseline  acc mean±std: {summary['baseline_all_features_accuracy']['mean']:.4f} ± {summary['baseline_all_features_accuracy']['std']:.4f}")
        print(f"Selected features mean±std: {summary['selected_feature_count']['mean']:.2f} ± {summary['selected_feature_count']['std']:.2f} (total={summary['total_features']})")
        if w.get("p_value") is not None:
            print(f"Wilcoxon p-value (paired): {w['p_value']:.6g} (n_used={w['n_used']})")
        else:
            print(f"Wilcoxon: {w.get('warning', 'not available')}")
        print(f"Figures: {summary['files']}")

    out_all = outdir / "analysis_summary_all_datasets.json"
    with out_all.open("w", encoding="utf-8") as f:
        json.dump(all_summaries, f, indent=2)


if __name__ == "__main__":
    main()


