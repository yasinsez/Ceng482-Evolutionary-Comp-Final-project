"""
Data loading utilities for PSO-SVM feature selection experiments.

Datasets are expected to live under:
    Datasets/

Supported datasets (to be implemented):
- Breast Cancer Wisconsin Diagnostic (WDBC)
- Wine (UCI Wine dataset)
- Connectionist Bench Sonar (Mines vs Rocks)
"""

from typing import Tuple

from pathlib import Path

import numpy as np


def load_wdbc_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Breast Cancer Wisconsin Diagnostic (WDBC) dataset.

    Returns
    -------
    X : np.ndarray, shape (n_samples, n_features)
        Feature matrix.
    y : np.ndarray, shape (n_samples,)
        Class labels.

    Notes
    -----
    This is a stub. Implement the actual file loading logic using:
    `Datasets/breast+cancer+wisconsin+diagnostic/wdbc.data`.
    """
    data_path = (
        Path(__file__).resolve().parent.parent
        / "Datasets"
        / "breast+cancer+wisconsin+diagnostic"
        / "wdbc.data"
    )

    with data_path.open("r", encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f if line.strip()]

    # Format: ID, Diagnosis(M/B), 30 real-valued features
    X = np.asarray([r[2:32] for r in rows], dtype=float)
    y = np.asarray([r[1] for r in rows], dtype=str)
    return X, y


def load_wine_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Wine dataset (UCI Wine, 3 classes, 13 features).

    Returns
    -------
    X : np.ndarray
        Feature matrix.
    y : np.ndarray
        Class labels.

    Notes
    -----
    This is a stub. Implement the actual file loading logic using:
    `Datasets/wine/wine.data`.

    The file format corresponds to the classic UCI Wine dataset described
    in `wine.names`.
    """
    data_path = (
        Path(__file__).resolve().parent.parent
        / "Datasets"
        / "wine"
        / "wine.data"
    )

    with data_path.open("r", encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f if line.strip()]

    # Format: class_label, 13 real-valued features
    y = np.asarray([r[0] for r in rows], dtype=int)
    X = np.asarray([r[1:] for r in rows], dtype=float)
    return X, y


def load_sonar_dataset() -> Tuple[np.ndarray, np.ndarray]:
    """
    Load the Connectionist Bench Sonar (Mines vs Rocks) dataset.

    Returns
    -------
    X : np.ndarray
        Feature matrix (typically 60 attributes per sample).
    y : np.ndarray
        Class labels (e.g., 'M' for mine and 'R' for rock, or encoded).

    Notes
    -----
    This is a stub. Implement the actual file loading logic using:
    `Datasets/connectionist+bench+sonar+mines+vs+rocks/sonar.all-data`.

    See also: `sonar.names` in the same folder for attribute and label
    descriptions (as in the UCI repository).
    """
    data_path = (
        Path(__file__).resolve().parent.parent
        / "Datasets"
        / "connectionist+bench+sonar+mines+vs+rocks"
        / "sonar.all-data"
    )

    with data_path.open("r", encoding="utf-8") as f:
        rows = [line.strip().split(",") for line in f if line.strip()]

    # Format: 60 real-valued features, label in last column ("R" or "M")
    X = np.asarray([r[:-1] for r in rows], dtype=float)
    y = np.asarray([r[-1] for r in rows], dtype=str)
    return X, y


def get_dataset(
    name: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generic helper to get a dataset by name.

    Parameters
    ----------
    name : {"wdbc", "wine", "sonar"}
        Name of the dataset.

    Returns
    -------
    X : np.ndarray
    y : np.ndarray

    Raises
    ------
    ValueError
        If an unsupported dataset name is given.
    """
    name = name.lower()
    if name == "wdbc":
        return load_wdbc_dataset()
    if name in ("wine", "wine_uci"):
        return load_wine_dataset()
    if name in ("sonar", "sonar_mines_rocks", "sonar-mines-rocks"):
        return load_sonar_dataset()

    raise ValueError(f"Unsupported dataset name: {name!r}")


