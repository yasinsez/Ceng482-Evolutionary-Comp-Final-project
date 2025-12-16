"""
Configuration helpers and default hyperparameters for PSO-SVM feature selection.

The specific values (e.g., C and RBF kernel parameters) should be chosen
following the PSO-SVM paper:
`Final Project/Papers/Feature Selection using PSO-SVM.pdf`.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class SVMConfig:
    """
    Configuration for the SVM classifier used as the PSO fitness evaluator.

    Attributes
    ----------
    C : float
        Regularization parameter of the SVM.
    kernel : str
        Kernel type to be used in the SVM (e.g., "rbf").
    gamma : Optional[float]
        Kernel coefficient for 'rbf', 'poly' and 'sigmoid'.
        If None, use the library default.
    r : Optional[float]
        Paper parameter name for the RBF kernel coefficient. In scikit-learn,
        this corresponds to `gamma`. If both `gamma` and `r` are provided,
        `gamma` takes precedence.
    """

    # Paper default: C = 2^12 = 4096
    C: float = 4096.0
    kernel: str = "rbf"
    gamma: Optional[float] = None
    r: Optional[float] = None


@dataclass
class PSOConfig:
    """
    Configuration for the PSO feature selection algorithm.

    Attributes
    ----------
    num_particles : int
        Number of particles in the swarm.
    num_iterations : int
        Maximum number of iterations (generations).
    inertia_weight : float
        Inertia weight (w) in the velocity update equation.
    cognitive_coeff : float
        Cognitive coefficient (c1) for the personal best component.
    social_coeff : float
        Social coefficient (c2) for the global best component.
    """

    num_particles: int = 30
    num_iterations: int = 50
    inertia_weight: float = 0.7
    cognitive_coeff: float = 1.5
    social_coeff: float = 1.5


def get_default_svm_config() -> SVMConfig:
    """
    Get a default SVM configuration.

    Notes
    -----
    Defaults are set to match the paper where possible:
    - C = 2^12 = 4096
    - `r` (paper) maps to `gamma` (scikit-learn). Set either `gamma` or `r`.
    See: `file:///Users/yasinsezgin/Downloads/Ceng/482/Final%20Project/Papers/Feature%20Selection%20using%20PSO-SVM.pdf`.
    """
    return SVMConfig()


def get_default_pso_config() -> PSOConfig:
    """
    Get a default PSO configuration for feature selection.

    Notes
    -----
    Stub: you may later tune these hyperparameters using the experimental
    setup described in the PSO-SVM paper.
    """
    return PSOConfig()


