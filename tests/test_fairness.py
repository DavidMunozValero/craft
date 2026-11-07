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
        expected = (100**2) / (4 * (0 + 0 + 0 + 100**2))
        assert idx == pytest.approx(expected, abs=1e-4)

    def test_empty_returns_one(self):
        assert FairnessMetrics().jain_index([]) == 1.0

    def test_all_zero_returns_zero(self):
        assert FairnessMetrics().jain_index([0, 0, 0]) == 0.0

    def test_two_values(self):
        idx = FairnessMetrics().jain_index([1, 3])
        expected = (4**2) / (2 * (1 + 9))
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
        fairness, ratios = FairnessMetrics.jains_fairness_index(
            scheduled, capacities, revenue
        )
        assert fairness == pytest.approx(1.0)

    def test_none_scheduled_returns_one(self):
        scheduled = np.array([False, False])
        capacities = {"ru1": 100, "ru2": 100}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
        }
        fairness, _ = FairnessMetrics.jains_fairness_index(
            scheduled, capacities, revenue
        )
        assert fairness == 1.0

    def test_partial_scheduled(self):
        scheduled = np.array([True, False, True])
        capacities = {"ru1": 50, "ru2": 50}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
            "s3": {"ru": "ru1", "importance": 1.0},
        }
        fairness, scheduled_sum = FairnessMetrics.jains_fairness_index(
            scheduled, capacities, revenue
        )
        assert 0.0 <= fairness <= 1.0
        assert "ru1" in scheduled_sum
        assert scheduled_sum["ru1"] == pytest.approx(2.0)

    def test_alpha_default_is_plain_jain(self):
        scheduled = np.array([True, True, False])
        capacities = {"ru1": 50.0, "ru2": 50.0}
        revenue = {
            "s1": {"ru": "ru1", "importance": 0.6},
            "s2": {"ru": "ru2", "importance": 0.4},
            "s3": {"ru": "ru2", "importance": 0.6},
        }
        default = FairnessMetrics.jains_fairness_index(scheduled, capacities, revenue)
        plain = FairnessMetrics.jains_fairness_index(
            scheduled, capacities, revenue, alpha=1.0
        )
        assert default[0] == pytest.approx(plain[0])

    def test_weighted_alpha_matches_transformed_ratios(self):
        """alpha raises the capacity ratios before the Jain formula."""
        scheduled = np.array([True, True, False, False])
        capacities = {"ru1": 50.0, "ru2": 50.0}
        revenue = {
            "s1": {"ru": "ru1", "importance": 0.9},
            "s2": {"ru": "ru2", "importance": 0.2},
            "s3": {"ru": "ru1", "importance": 0.1},
            "s4": {"ru": "ru2", "importance": 0.8},
        }
        alpha = 25.0
        fairness, _ = FairnessMetrics.jains_fairness_index(
            scheduled, capacities, revenue, alpha=alpha
        )
        ratios = np.array([0.9 / 50.0, 0.2 / 50.0])
        weighted = ratios**alpha
        expected = weighted.sum() ** 2 / (2 * (weighted**2).sum())
        assert fairness == pytest.approx(expected)

    def test_higher_alpha_penalizes_imbalance(self):
        scheduled = np.array([True, True])
        capacities = {"ru1": 75.0, "ru2": 25.0}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
        }
        mild = FairnessMetrics.jains_fairness_index(
            scheduled, capacities, revenue, alpha=1.0
        )[0]
        harsh = FairnessMetrics.jains_fairness_index(
            scheduled, capacities, revenue, alpha=25.0
        )[0]
        assert harsh < mild


class TestGiniFairnessIndex:
    def make_inputs(self):
        scheduled = np.array([True, True, False])
        capacities = {"ru1": 50.0, "ru2": 50.0}
        revenue = {
            "s1": {"ru": "ru1", "importance": 0.5},
            "s2": {"ru": "ru2", "importance": 0.5},
            "s3": {"ru": "ru2", "importance": 0.5},
        }
        return scheduled, capacities, revenue

    def test_equal_ratios_are_perfectly_fair(self):
        fairness, _ = FairnessMetrics.gini_fairness_index(
            *self.make_inputs(), alpha=10.0
        )
        assert fairness == pytest.approx(1.0)

    def unequal_inputs(self):
        """Inputs where both RUs receive scheduled services, unequally."""
        scheduled, capacities, revenue = self.make_inputs()
        revenue["s1"]["importance"] = 0.7
        revenue["s2"]["importance"] = 0.3
        revenue["s3"]["importance"] = 0.7
        return scheduled, capacities, revenue

    def test_unequal_ratios_reduce_fairness(self):
        scheduled, capacities, revenue = self.unequal_inputs()
        fairness, ratios = FairnessMetrics.gini_fairness_index(
            scheduled, capacities, revenue, alpha=10.0
        )
        assert 0.0 <= fairness < 1.0
        assert set(ratios) == set(capacities)

    def test_higher_alpha_magnifies_inequality(self):
        scheduled, capacities, revenue = self.unequal_inputs()
        mild = FairnessMetrics.gini_fairness_index(
            scheduled, capacities, revenue, alpha=1.0
        )[0]
        harsh = FairnessMetrics.gini_fairness_index(
            scheduled, capacities, revenue, alpha=10.0
        )[0]
        assert harsh < mild

    def test_empty_schedule_is_rejected(self):
        scheduled = np.array([False, False])
        capacities = {"ru1": 50.0, "ru2": 50.0}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
        }
        with pytest.raises(ValueError):
            FairnessMetrics.gini_fairness_index(scheduled, capacities, revenue)


class TestAtkinsonFairnessIndex:
    def make_inputs(self):
        scheduled = np.array([True, True, False])
        capacities = {"ru1": 50.0, "ru2": 50.0}
        revenue = {
            "s1": {"ru": "ru1", "importance": 0.5},
            "s2": {"ru": "ru2", "importance": 0.5},
            "s3": {"ru": "ru2", "importance": 0.5},
        }
        return scheduled, capacities, revenue

    def unequal_inputs(self):
        """Inputs where both RUs receive scheduled services, unequally."""
        scheduled, capacities, revenue = self.make_inputs()
        revenue["s1"]["importance"] = 0.7
        revenue["s2"]["importance"] = 0.3
        revenue["s3"]["importance"] = 0.7
        return scheduled, capacities, revenue

    def test_equal_ratios_are_perfectly_fair(self):
        fairness, _ = FairnessMetrics.atkinson_fairness_index(
            *self.make_inputs(), alpha=25.0
        )
        assert fairness == pytest.approx(1.0)

    def test_unequal_ratios_reduce_fairness(self):
        scheduled, capacities, revenue = self.unequal_inputs()
        fairness, ratios = FairnessMetrics.atkinson_fairness_index(
            scheduled, capacities, revenue, alpha=25.0
        )
        assert 0.0 <= fairness < 1.0
        assert set(ratios) == set(capacities)

    def test_empty_schedule_returns_one(self):
        scheduled = np.array([False, False])
        capacities = {"ru1": 50.0, "ru2": 50.0}
        revenue = {
            "s1": {"ru": "ru1", "importance": 1.0},
            "s2": {"ru": "ru2", "importance": 1.0},
        }
        fairness, _ = FairnessMetrics.atkinson_fairness_index(
            scheduled, capacities, revenue
        )
        assert fairness == 1.0
