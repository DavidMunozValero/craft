"""Gravitational field primitives for the GSA.

Functions computing the agent masses from fitness, the (possibly chaotic)
gravitational constant schedule and the gravitational field/acceleration
induced by the k-best agents on the rest of the population.
"""

from functools import lru_cache
from typing import Tuple

import numpy as np
from scipy.spatial.distance import euclidean, hamming


def mass_calculation(fit: np.ndarray) -> np.ndarray:
    """Compute the normalized mass of each agent from its fitness."""
    f_min, f_max = fit.min(), fit.max()
    if f_max == f_min:
        return np.ones(fit.shape) / len(fit)
    normalized_fit = (fit - f_min) / (f_max - f_min)
    mass = normalized_fit / normalized_fit.sum()
    return mass


def g_bin_constant(curr_iter: int, max_iters: int, g_zero: float = 1) -> float:
    """Linearly decreasing gravitational constant for the discrete space."""
    return g_zero * (1 - (curr_iter / max_iters))


def g_real_constant(curr_iter: int, max_iters: int, alpha: float = 20, g_zero: float = 100) -> float:
    """Exponentially decreasing gravitational constant for the real space."""
    return g_zero * np.exp(- ((alpha * curr_iter) / max_iters))


@lru_cache(maxsize=None)
def compute_x(i: int) -> float:
    """Sinusoidal chaotic map used to perturb the gravitational constant."""
    if i == 0:
        return 0.7
    prev_x = compute_x(i - 1)
    return 2.3 * prev_x ** 2 * np.sin(np.pi * prev_x)


def sin_chaotic_term(curr_iter: int, value: float) -> Tuple[float, float]:
    """Apply the sinusoidal chaotic term to ``value`` at the given iteration."""
    x = compute_x(curr_iter)
    return x * value, x


def g_field(
    population_size: int,
    dim: int,
    pos: np.ndarray,
    mass: np.ndarray,
    current_iter: int,
    max_iters: int,
    gravity_constant: float,
    r_power: int,
    elitist_check: bool,
    real: bool
) -> np.ndarray:
    """Calculate the gravitational field (acceleration) acting on agents.

    Only the ``k_best`` agents (heaviest masses) attract the rest. The
    distance is Euclidean for the real space and Hamming for the discrete
    space. A random factor perturbs each component as in the original GSA.
    """
    if dim == 0:
        return np.array([])

    pos = np.asarray(pos, dtype=float)
    if pos.size == 0:
        return np.array([])

    if pos.ndim == 1:
        if pos.shape[0] == population_size * dim:
            pos = pos.reshape(population_size, dim)
        elif pos.shape[0] == dim:
            pos = pos.reshape(1, dim)
        else:
            raise ValueError(f"Cannot reshape pos of shape {pos.shape} to ({population_size}, {dim})")
    if not dim > 0:
        return np.array([])

    final_per = 2
    if elitist_check:
        k_best = final_per + (1 - current_iter / max_iters) * (100 - final_per)
        k_best = round(population_size * k_best / 100)
    else:
        k_best = population_size

    ds = sorted(range(len(mass)), key=lambda k: mass[k], reverse=True)

    acc = np.zeros((population_size, dim))

    for r in range(population_size):
        for ii in range(k_best):
            z = ds[ii]
            if z != r:
                x = pos[r, :]
                y = pos[z, :]
                if real:
                    radius = euclidean(x, y)
                else:
                    radius = hamming(x, y)

                n = np.random.random(dim)
                acc[r, :] += n * gravity_constant * (mass[z] / (radius + np.finfo(float).eps)) * (pos[z, :] - pos[r, :])

    return acc
