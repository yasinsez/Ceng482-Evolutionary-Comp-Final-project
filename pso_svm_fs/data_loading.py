"""
Data loading utilities for PSO-SVM feature selection experiments.

Datasets are expected to live under:
    Final Project/Datasets/

Supported datasets (to be implemented):
- Breast Cancer Wisconsin Diagnostic (WDBC)
- Wine (UCI Wine dataset)
- Connectionist Bench Sonar (Mines vs Rocks)
"""

from typing import Tuple

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
    `Final Project/Datasets/breast+cancer+wisconsin+diagnostic/wdbc.data`.
    """

    with open("Final Project/Datasets/breast+cancer+wisconsin+diagnostic/wdbc.data", "r") as file:
        data = file.readlines()
    X = np.array([line.split(",")[2:32] for line in data])
    y = np.array([line.split(",")[1] for line in data])
    print(X.shape, y.shape)
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
    `Final Project/Datasets/wine/wine.data`.

    The file format corresponds to the classic UCI Wine dataset described
    in `wine.names`.
    """
    raise NotImplementedError("load_wine_dataset is not implemented yet.")


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
    `Final Project/Datasets/connectionist+bench+sonar+mines+vs+rocks/sonar.all-data`.

    See also: `sonar.names` in the same folder for attribute and label
    descriptions (as in the UCI repository).
    """
    raise NotImplementedError("load_sonar_dataset is not implemented yet.")


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


