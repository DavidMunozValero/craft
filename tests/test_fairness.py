"""Tests for craft.fairness: Jain, Gini, Atkinson indices."""

import numpy as np
import pytest

from craft.fairness import FairnessMetrics


class TestJainIndex:
    def test_perfect_equality(self):
        idx = FairnessMetrics().jain_index([10, 10, 10, 10])
        assert idx == pytest.approx(1.0)

    def test_maximum_inequality(self):
        idx = FairnessMetrics().jain_index([0, 0, 0, 100])
        expected = (100 ** 2) / (4 * (0 + 0 + 0 + 100 ** 2))
        assert idx == pytest.approx(expected, abs=1e-4)

    def test_empty_returns_one(self):
        assert FairnessMetrics().jain_index([]) == 1.0

    def test_all_zero_returns_zero(self):
        assert FairnessMetrics().jain_index([0, 0, 0]) == 0.0

    def test_two_values(self):
        idx = FairnessMetrics().jain_index([1, 3])
        expected = (4 ** 2) / (2 * (1 + 9))
        assert idx == pytest.approx(expected)


class TestGiniCoefficient:
    def test_perfect_equality(self):
        gini = FairnessMetrics().gini_coefficient([10, 10, 10, 10])
        assert gini == pytest.approx(0.0, abs=1e-6)

    def test_empty_returns_zero(self):
        assert FairnessMetrics().gini_coefficient([]) == 0.0

    def test_all_zero_returns_zero(self):
        assert FairnessMetrics().gini_coefficient([0, 0, 0]) == 0.0

    def test_increasing_inequality(self):
        fm = FairnessMetrics()
        gini_equal = fm.gini_coefficient([10, 10, 10])
        gini_unequal = fm.gini_coefficient([0, 10, 20])
        assert gini_unequal > gini_equal


class TestAtkinsonIndex:
    def test_perfect_equality(self):
        atk = FairnessMetrics().atkinson_index([10, 10, 10, 10], epsilon=1.0)
        assert atk == pytest.approx(0.0, abs=1e-6)

    def test_empty_returns_zero(self):
        assert FairnessMetrics().atkinson_index([]) == 0.0

    def test_all_zero_returns_zero(self):
        assert FairnessMetrics().atkinson_index([0, 0, 0]) == 0.0


class TestJainsFairnessIndex:
    def test_all_scheduled_equal_capacity(self):
        scheduled = np.array([True, True, True])
        capacities = {"ru1": 100, "ru2": 100, "ru3": 100}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
            "s3": {"ru": "ru3", "importance": 1.0},
        }
        fairness, ratios = FairnessMetrics.jains_fairness_index(scheduled, capacities, revenue)
        assert fairness == pytest.approx(1.0)

    def test_none_scheduled_returns_one(self):
        scheduled = np.array([False, False])
        capacities = {"ru1": 100, "ru2": 100}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
        }
        fairness, _ = FairnessMetrics.jains_fairness_index(scheduled, capacities, revenue)
        assert fairness == 1.0

    def test_partial_scheduled(self):
        scheduled = np.array([True, False, True])
        capacities = {"ru1": 50, "ru2": 50}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
            "s3": {"ru": "ru1", "importance": 1.0},
        }
        fairness, scheduled_sum = FairnessMetrics.jains_fairness_index(scheduled, capacities, revenue)
        assert 0.0 <= fairness <= 1.0
        assert "ru1" in scheduled_sum
        assert scheduled_sum["ru1"] == pytest.approx(2.0)
