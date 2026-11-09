"""Tests for the common solution evaluator (``craft.ga.evaluation``)."""

from pathlib import Path

import numpy as np
import pytest

from craft.ga import GATimetabling, SolutionEvaluator, inequity_percent


@pytest.fixture()
def problem(supply, revenue_behavior) -> GATimetabling:
    """Timetabling problem over the session fixture supply."""
    return GATimetabling(supply, revenue_behavior)


@pytest.fixture()
def evaluator(problem) -> SolutionEvaluator:
    return SolutionEvaluator(problem, conflict_penalty=2.0)


def full_mask(problem) -> np.ndarray:
    return np.ones(problem.n_services, dtype=bool)


class TestInequityPercent:
    def test_proportional_allocation_is_zero(self, problem):
        """Scheduling everything gives every RU its full importance."""
        from craft.ga.objectives import compute_capacities

        capacities = compute_capacities(problem.revenue_behavior)
        assert inequity_percent(
            problem, capacities, full_mask(problem)
        ) == pytest.approx(0.0)

    def test_concentrated_allocation_is_positive(self, problem):
        from craft.ga.objectives import compute_capacities

        capacities = compute_capacities(problem.revenue_behavior)
        mask = np.zeros(problem.n_services, dtype=bool)
        mask[0] = True
        assert 0.0 < inequity_percent(problem, capacities, mask) <= 100.0


class TestSolutionEvaluator:
    def test_greedy_matches_the_published_baseline(self, problem, evaluator):
        """The greedy schedule is conflict-free and its raw revenue equals
        the canon sum of the kept services."""
        kept, canon_sum = problem.greedy_baseline()
        mask = np.array([sid in set(kept) for sid in problem.service_ids])
        result = evaluator.evaluate(problem.reference_departures.astype(float), mask)
        assert result["conflicts"] == 0
        assert result["penalized"] == pytest.approx(result["revenue"])
        assert result["revenue"] == pytest.approx(canon_sum)
        assert result["planned"] == len(kept)

    def test_planning_everything_keeps_consistency(self, problem, evaluator):
        """On the (possibly conflict-free) fixture instance the raw and
        penalized metrics stay consistent for the full mask."""
        result = evaluator.evaluate(
            problem.reference_departures.astype(float), full_mask(problem)
        )
        assert result["planned"] == problem.n_services
        assert result["revenue"] > 0.0
        assert result["penalized"] <= result["revenue"]
        assert result["conflicts"] >= 0

    @pytest.mark.skipif(
        not (
            Path(__file__).resolve().parent.parent
            / "data"
            / "synthetic"
            / "timetable.yaml"
        ).exists(),
        reason="synthetic data not generated (run notebook 0.0)",
    )
    def test_planning_everything_leaves_conflicts_on_synthetic(self):
        """On the conflicting 50-service instance the full mask leaves the
        103 requested-time conflicts and the penalized revenue is lower."""
        import yaml

        from robin.supply.entities import Supply

        synthetic = Path(__file__).resolve().parent.parent / "data" / "synthetic"
        supply = Supply.from_yaml(path=str(synthetic / "timetable.yaml"))
        revenue = yaml.safe_load((synthetic / "revenue.yaml").read_text())["services"]
        problem = GATimetabling(supply, revenue)
        evaluator = SolutionEvaluator(problem, conflict_penalty=2.0)
        result = evaluator.evaluate(
            problem.reference_departures.astype(float), full_mask(problem)
        )
        assert result["conflicts"] == 103
        assert result["penalized"] < result["revenue"]

    def test_empty_schedule_scores_zero_revenue(self, problem, evaluator):
        empty = np.zeros(problem.n_services, dtype=bool)
        result = evaluator.evaluate(problem.reference_departures.astype(float), empty)
        assert result["revenue"] == 0.0
        assert result["planned"] == 0
        assert result["conflicts"] == 0
        assert result["fairness_jain"] == 1.0
        assert result["inequity"] == 0.0

    def test_metrics_keys(self, problem, evaluator):
        result = evaluator.evaluate(
            problem.reference_departures.astype(float), full_mask(problem)
        )
        assert set(result) == {
            "revenue",
            "penalized",
            "planned",
            "conflicts",
            "fairness_jain",
            "fairness_gini",
            "fairness_atkinson",
            "inequity",
        }
