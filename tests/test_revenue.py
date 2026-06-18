"""Tests for craft.revenue: RevenueSimulator, RevenueCalculator, penalty_function."""

from math import cos, e, pi

import numpy as np
import pytest

from craft.revenue import RevenueCalculator, RevenueSimulator, penalty_function


class TestPenaltyFunction:
    def test_zero_deviation_zero_penalty(self):
        assert penalty_function(0.0, k=10) == pytest.approx(0.0, abs=1e-6)

    def test_positive_deviation_positive_penalty(self):
        assert penalty_function(0.5, k=10) > 0.0

    def test_penalty_bounded(self):
        for x in [0.1, 0.5, 1.0, 2.0]:
            for k in [0.01, 1, 10, 100]:
                p = penalty_function(x, k)
                assert 0.0 <= p <= 1.0 + 1e-6

    def test_formula_matches(self):
        x, k = 0.3, 5
        expected = 1 - e ** (-k * x ** 2) * (0.5 * cos(pi * x) + 0.5)
        assert penalty_function(x, k) == pytest.approx(expected)


class TestRevenueSimulator:
    def test_simulate_revenue_keys(self, supply):
        np.random.seed(42)
        rev = RevenueSimulator(supply=supply).simulate_revenue(alpha=2 / 3)
        assert len(rev) == len(supply.services)
        for sid, data in rev.items():
            assert "canon" in data
            assert "ru" in data
            assert "k" in data
            assert "dt_max_penalty" in data
            assert "tt_max_penalty" in data
            assert "importance" in data
            assert data["canon"] > 0
            assert 0.0 <= data["importance"] <= 1.0

    def test_importance_normalized_per_ru(self, supply):
        np.random.seed(42)
        rev = RevenueSimulator(supply=supply).simulate_revenue(alpha=2 / 3)
        ru_importances = {}
        for data in rev.values():
            ru = data["ru"]
            ru_importances.setdefault(ru, []).append(data["importance"])
        for ru, imps in ru_importances.items():
            assert sum(imps) == pytest.approx(1.0, abs=1e-6)


class TestRevenueCalculator:
    def test_compute_all_revenues(self, timetabling):
        rc = timetabling.revenue_calculator
        assert len(rc.service_revenues) == timetabling.n_services
        for sid, data in rc.service_revenues.items():
            assert "revenue" in data
            assert "canon" in data
            assert "importance" in data
            assert "ru" in data

    def test_recompute_after_update(self, timetabling):
        rc = timetabling.revenue_calculator
        original = {s: rc.service_revenues[s]["revenue"] for s in rc.service_revenues}
        rc.recompute_all_revenues()
        recomputed = {s: rc.service_revenues[s]["revenue"] for s in rc.service_revenues}
        for sid in original:
            assert recomputed[sid] == pytest.approx(original[sid])

    def test_compute_total_revenue(self, timetabling):
        rc = timetabling.revenue_calculator
        service_ids = list(rc.reference_schedules.keys())
        mask = np.array([True] * len(service_ids))
        total = rc.compute_total_revenue(
            service_ids, mask, timetabling.schedule_manager.is_service_feasible,
        )
        assert total >= 0
