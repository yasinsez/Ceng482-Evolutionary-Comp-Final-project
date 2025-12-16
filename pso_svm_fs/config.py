from dataclasses import dataclass
from typing import Optional

@dataclass
class SVMConfig:
    """
    SVM hyperparameters used in the fitness evaluation.

    Notes
    -----
    - The paper reports C = 2^12 = 4096.
    - The paper uses the symbol `r` for the RBF kernel coefficient; in scikit-learn
      this corresponds to `gamma`. In our code, `r` is mapped to `gamma` when `gamma`
      is not explicitly set.
    """

    # Paper default: C = 2^12 = 4096
    C: float = 4096.0
    kernel: str = "rbf"
    gamma: Optional[float] = None
    # Paper-style name for the RBF coefficient (mapped to sklearn's `gamma`).
    r: Optional[float] = 1.0

@dataclass
class PSOConfig:
    """
    PSO hyperparameters for binary feature selection.

    Notes
    -----
    Some PSO-SVM setups use c1=c2=2.0 as in many PSO references / paper tables.
    """

    num_particles: int = 30
    num_iterations: int = 50
    # Inertia weight (w). Some variants decrease w over time; we keep it fixed here.
    inertia_weight: float = 0.7
    # Cognitive and social coefficients (c1, c2).
    cognitive_coeff: float = 2.0
    social_coeff: float = 2.0
    v_min: float = -6.0
    v_max: float = 6.0
    init_position_prob: float = 0.5

def get_svm_config_for_dataset(dataset_name: str) -> SVMConfig:
    """
    Return dataset-specific SVM hyperparameters (from the paper's Table 1).

    The paper reports different `r` (RBF coefficient) values per dataset.
    We store them in `config.r` (paper name) and later map to sklearn's `gamma`.
    """
    config = SVMConfig()
    
    # All datasets in the paper use C = 2^12 = 4096
    config.C = 4096.0
    
    name = dataset_name.lower().strip()
    
    if name == "wdbc":
        # WDBC: r = 2^0 = 1
        config.r = 1.0
    elif name == "wine":
        # Wine: r = 2^4 = 16
        config.r = 16.0
    elif name == "ionosphere":
        # Ionosphere: r = 2^-1 = 0.5
        config.r = 0.5
    elif name == "sonar":
        # Sonar: r = 2^0 = 1
        config.r = 1.0
    elif name == "vowel":
        # Vowel: r = 2^4 = 16
        config.r = 16.0
    else:
        # Fallback for unknown datasets (WDBC settings are a safe default)
        print(
            f"Warning: no dataset-specific SVM config for {dataset_name!r}; using default r=1."
        )
        config.r = 1.0
        
    return config

def get_default_pso_config() -> PSOConfig:
    return PSOConfig()


def get_default_svm_config() -> SVMConfig:
    """
    Return a default SVM configuration.

    Notes
    -----
    This is a generic fallback (C=4096, r=1) when a dataset-specific configuration
    is not selected explicitly. Prefer `get_svm_config_for_dataset(dataset_name)`
    to match the paper's Table 1 values.
    """
    return SVMConfig()