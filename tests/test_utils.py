"""Tests for craft.utils: get_rus_revenue, is_better_solution."""

import pandas as pd

from craft.utils import get_rus_revenue, is_better_solution


class TestGetRusRevenue:
    def test_basic_aggregation(self, supply):
        df = pd.DataFrame(
            {
                "service": [s.id for s in supply.services[:4]],
                "price": [10.0, 20.0, 30.0, 40.0],
            }
        )
        revenue = get_rus_revenue(supply, df)
        assert isinstance(revenue, dict)
        assert len(revenue) > 0
        for tsp, rev in revenue.items():
            assert rev > 0


class TestIsBetterSolution:
    def test_empty_best_is_better(self):
        assert is_better_solution({"ru1": 100}, {}) is True

    def test_more_rus_is_better(self):
        assert is_better_solution({"ru1": 100, "ru2": 50}, {"ru1": 100}) is True

    def test_majority_improvement_is_better(self):
        current = {"ru1": 200, "ru2": 50, "ru3": 100}
        best = {"ru1": 100, "ru2": 100, "ru3": 50}
        assert is_better_solution(current, best) is True

    def test_no_improvement(self):
        current = {"ru1": 50, "ru2": 50}
        best = {"ru1": 100, "ru2": 100}
        assert is_better_solution(current, best) is False
