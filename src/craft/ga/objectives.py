"""Bi-objective (revenue + fairness) builders for the direct encoding.

The default single objective of the genetic backends is the penalized
revenue of the problem head. This module turns the same genome landscape
into the two-dimensional formulation consumed by the NSGA-II backend:
objective one stays the penalized revenue and objective two is a
capacity-aware fairness index over the RU allocation of the planned
services, following the importance-vs-capacity ratios of the Fair-oriented
timetabling paper (Jain/Gini/Atkinson on ratios raised to a sensitivity
exponent ``alpha``).

The capacity of a RU is its share of the requested services, and the
importance of a scheduled service is the weight declared by its RU in the
revenue behavior, so the fairness axis only depends on the planning mask
and is invariant to the departure-time deltas.
"""

from typing import Any, Callable, Dict, Mapping, Optional

import numpy as np

from ..fairness import FairnessMetrics
from .timetabling import GATimetabling

FAIRNESS_ALPHA = {"jain": 25.0, "gini": 10.0, "atkinson": 25.0}


def compute_capacities(
    revenue_behavior: Mapping[str, Mapping[str, Any]],
) -> Dict[str, float]:
    """Capacity share of every RU as the percentage of requested services.

    Args:
        revenue_behavior: Revenue parameters per service (must expose the
            ``ru`` of every service).

    Returns:
        Percentage of the requested services owned by each RU.
    """
    counts: Dict[str, int] = {}
    for params in revenue_behavior.values():
        ru = params["ru"]
        counts[ru] = counts.get(ru, 0) + 1
    total = sum(counts.values())
    return {ru: 100.0 * count / total for ru, count in counts.items()}


def revenue_fairness_objectives(
    problem: GATimetabling,
    index: str = "jain",
    alpha: Optional[float] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Build the ``[revenue, fairness]`` objective vector of a genome.

    Objective one is the penalized revenue of the problem head (the very
    landscape optimized by the GA); objective two is the capacity-aware
    fairness index of the planned services. The empty schedule scores
    ``(0, 0)`` so the null plan always stays dominated and off the Pareto
    front.

    Args:
        problem: Timetabling problem providing the genome decoding, the
            penalized revenue and the per-service revenue behavior.
        index: Fairness index, one of ``jain``, ``gini`` or ``atkinson``.
        alpha: Sensitivity exponent of the index; ``None`` uses the
            calibration of the Fair-oriented paper (:data:`FAIRNESS_ALPHA`:
            25 for Jain and Atkinson, 10 for Gini).

    Returns:
        Callable from genome to the objective vector (maximization), ready
        to be passed as the ``objectives`` argument of
        :class:`~craft.ga.nsga2.NSGA2`.

    Raises:
        ValueError: If ``index`` is not a known fairness index or ``alpha``
            is not positive.
    """
    if index not in FAIRNESS_ALPHA:
        raise ValueError(f"Unknown fairness index: {index!r}")
    if alpha is None:
        alpha = FAIRNESS_ALPHA[index]
    if alpha <= 0:
        raise ValueError("alpha must be positive")
    capacities = compute_capacities(problem.revenue_behavior)
    fairness_index = {
        "jain": FairnessMetrics.jains_fairness_index,
        "gini": FairnessMetrics.gini_fairness_index,
        "atkinson": FairnessMetrics.atkinson_fairness_index,
    }[index]

    def objectives(genome: np.ndarray) -> np.ndarray:
        """Objective vector of one genome: penalized revenue and fairness."""
        mask = problem.decode(genome).discrete
        revenue = problem.fitness(genome)
        if not mask.any():
            return np.array([revenue, 0.0])
        fairness, _ = fairness_index(
            mask, capacities, problem.revenue_behavior, alpha=alpha
        )
        return np.array([revenue, fairness])

    return objectives
