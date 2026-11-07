"""Common evaluation of decoded solutions across backends.

Every solver output — a departure vector plus a planning mask, whatever the
backend that produced it — is measured with the same ruler:

* operational feasibility filter: only services whose decoded operating
  times hold (the same filter the heuristic-mask methods apply internally)
  earn revenue and count towards the RU allocation;
* raw revenue of the surviving services;
* conflicts recomputed over the decoded schedule (the dynamic checker of
  the direct encoding, validated 1:1 against ``robin``) and the penalized
  revenue of the GA landscape (``revenue - lambda * sum min(rev_i, rev_j)``);
* capacity-aware fairness indices with the calibration of the Fair-oriented
  paper (Jain 25, Gini 10, Atkinson 25) and the pairwise-difference inequity
  of that paper (percent of the theoretical maximum).

This module is the measurement standard of the comparative experiments
(``scripts/run_comparison.py``, notebook 8.0).
"""

from itertools import combinations
from typing import Callable, Dict, Mapping, Tuple

import numpy as np

from ..fairness import FairnessMetrics
from .objectives import compute_capacities
from .timetabling import GATimetabling

FAIRNESS_EVALUATORS: Mapping[
    str, Tuple[Callable[..., Tuple[float, Mapping]], float]
] = {
    "jain": (FairnessMetrics.jains_fairness_index, 25.0),
    "gini": (FairnessMetrics.gini_fairness_index, 10.0),
    "atkinson": (FairnessMetrics.atkinson_fairness_index, 25.0),
}


def inequity_percent(
    problem: GATimetabling, capacities: Mapping[str, float], mask: np.ndarray
) -> float:
    """Inequity of an allocation as defined in the Fair-oriented paper.

    Sum of the absolute pairwise differences between the importances
    assigned to every RU, normalized by the theoretical maximum (``n**2 / 4``
    for an even number of RUs, ``(n // 2) * ((n + 1) // 2)`` otherwise) and
    expressed in percent.

    Args:
        problem: Timetabling problem providing the revenue behavior.
        capacities: Capacity share per RU (see ``compute_capacities``).
        mask: Boolean planning mask in ``problem.service_ids`` order.

    Returns:
        The inequity percent in ``[0, 100]``.
    """
    sums = FairnessMetrics.sum_importance(
        np.asarray(mask, dtype=bool), problem.revenue_behavior
    )
    values = [sums.get(ru, 0.0) for ru in capacities]
    pairwise = sum(abs(a - b) for a, b in combinations(values, 2))
    n = len(values)
    max_inequity = (n**2 / 4) if n % 2 == 0 else (n // 2) * ((n + 1) // 2)
    return 100.0 * pairwise / max_inequity


class SolutionEvaluator:
    """Measure (real departures, planning mask) solutions of any backend.

    Args:
        problem: The direct-encoding problem head providing the schedule
            manager, the revenue calculator and the dynamic conflict checker.
        conflict_penalty: Weight ``lambda`` of the penalized revenue.
    """

    def __init__(self, problem: GATimetabling, conflict_penalty: float = 2.0) -> None:
        self.problem = problem
        self.conflict_penalty = conflict_penalty
        self.capacities = compute_capacities(problem.revenue_behavior)

    def evaluate(self, real: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
        """Evaluate one solution with the common measurement standard.

        Args:
            real: Flat departure vector (any backend layout, which matches
                the problem layout).
            mask: Boolean planning mask in ``problem.service_ids`` order.

        Returns:
            Dictionary with ``revenue`` (raw, operative services only),
            ``penalized`` (raw minus ``lambda`` times the conflict penalty),
            ``planned`` (operative services), ``conflicts`` (residual pairs
            over the decoded schedule), ``fairness_jain/gini/atkinson`` and
            ``inequity``. Fairness indices are ``NaN`` when a capacity-aware
            index cannot be computed (no scheduled service of some RU for
            Gini/Atkinson) and 1.0 for the empty allocation.
        """
        problem = self.problem
        real = np.asarray(real)
        mask = np.asarray(mask, dtype=bool)
        problem.schedule_manager.update_from_solution(real)
        problem.revenue_calculator.updated_schedule = dict(
            problem.schedule_manager.updated_schedule
        )
        problem.revenue_calculator.recompute_all_revenues()
        feasible = np.array(
            [
                mask[i] and problem.schedule_manager.is_service_feasible(sid)
                for i, sid in enumerate(problem.service_ids)
            ]
        )
        planned = [sid for i, sid in enumerate(problem.service_ids) if feasible[i]]
        revenues = {
            sid: problem.revenue_calculator.get_service_revenue(sid) for sid in planned
        }
        revenue = float(sum(revenues.values()))
        rows = [problem.row_index[sid] for sid in planned]
        matrix = problem.conflict_matrix(
            dict(problem.schedule_manager.updated_schedule), planned
        )
        sub = matrix[np.ix_(rows, rows)] if rows else np.zeros((0, 0), dtype=bool)
        conflicts = int(sub.sum() // 2)
        penalty = float(
            sum(
                min(revenues[planned[a]], revenues[planned[b]])
                for a in range(len(planned))
                for b in range(a + 1, len(planned))
                if sub[a, b]
            )
        )
        result: Dict[str, float] = {
            "revenue": revenue,
            "penalized": revenue - self.conflict_penalty * penalty,
            "planned": int(feasible.sum()),
            "conflicts": conflicts,
        }
        scheduled_rus = {problem.revenue_behavior[sid]["ru"] for sid in planned}
        for name, (fn, alpha) in FAIRNESS_EVALUATORS.items():
            if not planned:
                result[f"fairness_{name}"] = 1.0
            elif name in ("gini", "atkinson") and len(scheduled_rus) < len(
                self.capacities
            ):
                result[f"fairness_{name}"] = float("nan")
            else:
                fair, _ = fn(
                    feasible, self.capacities, problem.revenue_behavior, alpha=alpha
                )
                result[f"fairness_{name}"] = float(fair)
        result["inequity"] = inequity_percent(problem, self.capacities, feasible)
        return result
