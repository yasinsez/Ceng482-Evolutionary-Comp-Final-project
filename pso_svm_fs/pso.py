"""
Particle Swarm Optimization (PSO) for feature selection.

This module implements a **binary PSO** for feature selection:

- Each particle position is a binary vector (0/1) indicating selected features.
- Velocities are continuous; positions are updated via sigmoid(velocity) -> probability.
"""

from __future__ import annotations

from typing import Callable, Dict, Tuple

import numpy as np

from .config import PSOConfig

FitnessFunction = Callable[[np.ndarray], float]


def _sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically-stable sigmoid."""
    x = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-x))


def initialize_swarm(
    num_particles: int,
    num_features: int,
    pso_config: PSOConfig,
    random_state: int | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Initialize positions and velocities of a binary PSO swarm.

    Returns
    -------
    positions : np.ndarray, shape (num_particles, num_features)
        Initial particle positions (binary feature masks).
    velocities : np.ndarray, shape (num_particles, num_features)
        Initial particle velocities (continuous).
    """
    if num_particles <= 0 or num_features <= 0:
        raise ValueError("num_particles and num_features must be positive.")

    rng = np.random.default_rng(random_state)

    positions = (
        rng.random((num_particles, num_features)) < pso_config.init_position_prob
    ).astype(np.int8)
    velocities = rng.uniform(
        low=pso_config.v_min, high=pso_config.v_max, size=(num_particles, num_features)
    ).astype(float)

    return positions, velocities


def update_velocities_and_positions(
    positions: np.ndarray,
    velocities: np.ndarray,
    personal_best_positions: np.ndarray,
    global_best_position: np.ndarray,
    pso_config: PSOConfig,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    One binary PSO update step:

    v = w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x = 1 if rand() < sigmoid(v) else 0
    """
    r1 = rng.random(size=velocities.shape)
    r2 = rng.random(size=velocities.shape)

    new_velocities = (
        pso_config.inertia_weight * velocities
        + pso_config.cognitive_coeff * r1 * (personal_best_positions - positions)
        + pso_config.social_coeff * r2 * (global_best_position - positions)
    )
    new_velocities = np.clip(new_velocities, pso_config.v_min, pso_config.v_max)

    probs = _sigmoid(new_velocities)
    new_positions = (rng.random(size=probs.shape) < probs).astype(np.int8)

    return new_positions, new_velocities


def run_pso_feature_selection(
    num_features: int,
    fitness_fn: FitnessFunction,
    pso_config: PSOConfig,
    random_state: int | None = None,
) -> Dict[str, np.ndarray | float]:
    """
    Run global-best (gbest) binary PSO to maximize `fitness_fn`.

    Returns
    -------
    results : dict with keys:
      - best_position: np.ndarray (num_features,) best binary mask
      - best_fitness: float
      - fitness_history: np.ndarray (num_iterations,) best fitness per iteration
    """
    if num_features <= 0:
        raise ValueError("num_features must be positive.")

    rng = np.random.default_rng(random_state)

    positions, velocities = initialize_swarm(
        num_particles=pso_config.num_particles,
        num_features=num_features,
        pso_config=pso_config,
        random_state=random_state,
    )

    # Initial fitness
    fitness_values = np.asarray(
        [fitness_fn(positions[i]) for i in range(pso_config.num_particles)], dtype=float
    )

    personal_best_positions = positions.copy()
    personal_best_fitness = fitness_values.copy()

    best_idx = int(np.argmax(personal_best_fitness))
    global_best_position = personal_best_positions[best_idx].copy()
    global_best_fitness = float(personal_best_fitness[best_idx])

    fitness_history = np.empty(pso_config.num_iterations, dtype=float)

    for t in range(pso_config.num_iterations):
        fitness_history[t] = global_best_fitness

        positions, velocities = update_velocities_and_positions(
            positions=positions,
            velocities=velocities,
            personal_best_positions=personal_best_positions,
            global_best_position=global_best_position,
            pso_config=pso_config,
            rng=rng,
        )

        fitness_values = np.asarray(
            [fitness_fn(positions[i]) for i in range(pso_config.num_particles)], dtype=float
        )

        improved = fitness_values > personal_best_fitness
        if np.any(improved):
            personal_best_fitness[improved] = fitness_values[improved]
            personal_best_positions[improved] = positions[improved]

        best_idx = int(np.argmax(personal_best_fitness))
        if float(personal_best_fitness[best_idx]) > global_best_fitness:
            global_best_fitness = float(personal_best_fitness[best_idx])
            global_best_position = personal_best_positions[best_idx].copy()

    return {
        "best_position": global_best_position.astype(np.int8),
        "best_fitness": float(global_best_fitness),
        "fitness_history": fitness_history,
    }


