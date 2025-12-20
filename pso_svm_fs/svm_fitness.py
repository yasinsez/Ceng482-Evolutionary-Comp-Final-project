"""
SVM-based fitness evaluation for PSO feature selection.

The PSO-SVM paper uses SVMs with a one-versus-rest strategy as the
fitness function for feature selection:
`/Papers/Feature%20Selection%20using%20PSO-SVM.pdf`.
"""

from typing import Optional

import numpy as np

from .config import SVMConfig

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

def evaluate_feature_subset_with_svm(
    X: np.ndarray,
    y: np.ndarray,
    feature_mask: np.ndarray,
    svm_config: Optional[SVMConfig] = None,
    num_folds: int = 10,
    random_state: Optional[int] = None,
    n_jobs: int = -1,
) -> float:
    """
    Evaluate a binary feature subset using an SVM classifier with cross-validation.

    Parameters
    ----------
    X : np.ndarray, shape (n_samples, n_features)
        Full feature matrix.
    y : np.ndarray, shape (n_samples,)
        Class labels.
    feature_mask : np.ndarray, shape (n_features,)
        Binary vector indicating which features are selected (1) or not (0).
    svm_config : SVMConfig, optional
        Hyperparameters for the SVM; if None, use `get_default_svm_config()`.
    num_folds : int, default 10
        Number of cross-validation folds (e.g., 10-fold CV as in the paper).
    random_state : int, optional
        Random seed for reproducibility.

    Returns
    -------
    fitness : float
        Fitness value, typically the average classification accuracy over the
        cross-validation folds.

    Notes
    -----
    Uses `sklearn.svm.SVC` as the SVM classifier. To align with the paper's
    multiclass setup, we wrap it with `OneVsRestClassifier` (works for binary too).
    Scaling is included inside the CV pipeline to avoid data leakage.
    """
    if num_folds <= 1:
        raise ValueError("num_folds must be >= 2.")

    svm_config = svm_config or SVMConfig()

    # Ensure feature_mask is a 1D boolean mask of length n_features.
    feature_mask = np.asarray(feature_mask).ravel().astype(bool)
    if feature_mask.shape[0] != X.shape[1]:
        raise ValueError(
            f"feature_mask length ({feature_mask.shape[0]}) must match X.shape[1] ({X.shape[1]})."
        )

    # If no features are selected, define worst fitness.
    if not np.any(feature_mask):
        return 0.0

    x_selected = X[:, feature_mask]

    # Paper uses parameter name `r` for the RBF coefficient; scikit-learn calls it `gamma`.
    gamma_value = svm_config.gamma if svm_config.gamma is not None else svm_config.r
    base_svc = SVC(kernel=svm_config.kernel, C=svm_config.C, gamma=gamma_value)
    clf = OneVsRestClassifier(base_svc)

    # Pipeline for scaling and classification (to avoid data leakage).
    # StandardScaler is used to standardize the features before classification.
    # Data leakage is when the model is trained on the test data or the data that is not available at the time of prediction.
    # This is a common problem in machine learning and it is important to avoid it.
    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("clf", clf),
        ]
    )

    cv = StratifiedKFold(n_splits=num_folds, shuffle=True, random_state=random_state)
    scores = cross_val_score(model, x_selected, y, cv=cv, scoring="accuracy", n_jobs=n_jobs)
    return float(np.mean(scores))


