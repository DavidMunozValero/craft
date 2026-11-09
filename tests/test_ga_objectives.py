"""Tests for the bi-objective (revenue + fairness) builders."""

import numpy as np
import pytest

from craft.fairness import FairnessMetrics
from craft.ga import (
    GATimetabling,
    compute_capacities,
    revenue_fairness_objectives,
)
from craft.ga.objectives import FAIRNESS_ALPHA


@pytest.fixture()
def problem(supply, revenue_behavior) -> GATimetabling:
    """Timetabling problem over the session fixture supply."""
    return GATimetabling(supply, revenue_behavior)


def planned_genome(problem: GATimetabling, mask: np.ndarray) -> np.ndarray:
    """Zero-delta genome with the given planned mask."""
    genome = np.zeros(problem.genome_size, dtype=np.int64)
    for i, block in enumerate(problem.slices):
        genome[block.start] = int(mask[i])
    return genome


class TestComputeCapacities:
    def test_capacities_are_percentages(self, problem):
        """Each RU share is its percentage of the requested services."""
        counts: dict = {}
        for params in problem.revenue_behavior.values():
            counts[params["ru"]] = counts.get(params["ru"], 0) + 1
        capacities = compute_capacities(problem.revenue_behavior)
        total = problem.n_services
        assert capacities == {
            ru: pytest.approx(100.0 * count / total) for ru, count in counts.items()
        }

    def test_capacities_sum_to_hundred(self, problem):
        capacities = compute_capacities(problem.revenue_behavior)
        assert sum(capacities.values()) == pytest.approx(100.0)


class TestRevenueFairnessObjectives:
    def test_objectives_match_fitness_and_jain(self, problem):
        """Objective one is the penalized revenue and objective two the
        capacity-aware Jain index over the planning mask."""
        objectives = revenue_fairness_objectives(problem, index="jain", alpha=25.0)
        mask = np.ones(problem.n_services, dtype=bool)
        genome = planned_genome(problem, mask)
        values = objectives(genome)

        capacities = compute_capacities(problem.revenue_behavior)
        expected_fairness, _ = FairnessMetrics.jains_fairness_index(
            mask, capacities, problem.revenue_behavior, alpha=25.0
        )
        assert values.shape == (2,)
        assert values[0] == pytest.approx(problem.fitness(genome))
        assert values[1] == pytest.approx(expected_fairness)
        assert 0.0 <= values[1] <= 1.0

    def test_empty_schedule_scores_zero(self, problem):
        """The null plan is scored (0, 0) so it never enters the front."""
        objectives = revenue_fairness_objectives(problem)
        genome = planned_genome(problem, np.zeros(problem.n_services, dtype=bool))
        values = objectives(genome)
        assert values[0] == 0.0
        assert values[1] == 0.0

    def test_alpha_none_follows_calibration(self, problem):
        """Without an explicit exponent each index uses its calibrated one."""
        mask = np.ones(problem.n_services, dtype=bool)
        genome = planned_genome(problem, mask)
        for index, alpha in FAIRNESS_ALPHA.items():
            default = revenue_fairness_objectives(problem, index=index)(genome)
            explicit = revenue_fairness_objectives(problem, index=index, alpha=alpha)(
                genome
            )
            assert np.array_equal(default, explicit)

    def test_gini_and_atkinson_are_supported(self, problem):
        """Every calibrated index yields a two-dimensional bounded vector."""
        genome = planned_genome(problem, np.ones(problem.n_services, dtype=bool))
        for index in ("gini", "atkinson"):
            values = revenue_fairness_objectives(problem, index=index)(genome)
            assert values.shape == (2,)
            assert 0.0 <= values[1] <= 1.0

    def test_unknown_index_is_rejected(self, problem):
        with pytest.raises(ValueError):
            revenue_fairness_objectives(problem, index="theil")

    def test_non_positive_alpha_is_rejected(self, problem):
        with pytest.raises(ValueError):
            revenue_fairness_objectives(problem, alpha=0.0)
