"""Fairness metrics for railway scheduling."""

import numpy as np
from typing import Mapping, Tuple, Any


class FairnessMetrics:
    """Calculate fairness indices for RU capacity allocation."""

    def jain_index(self, revenue_values: list) -> float:
        """Calculate simple Jain index from revenue values."""
        n = len(revenue_values)
        if n == 0:
            return 1.0
        total = sum(revenue_values)
        if total == 0:
            return 0.0
        sum_squared = sum(x ** 2 for x in revenue_values)
        return (total ** 2) / (n * sum_squared) if sum_squared > 0 else 0.0

    def gini_coefficient(self, revenue_values: list) -> float:
        """Calculate Gini coefficient from revenue values."""
        n = len(revenue_values)
        if n == 0:
            return 0.0
        sorted_vals = sorted(revenue_values)
        cumsum = 0
        for i, val in enumerate(sorted_vals):
            cumsum += (i + 1) * val
        total = sum(sorted_vals)
        if total == 0:
            return 0.0
        return (2 * cumsum) / (n * total) - (n + 1) / n

    def atkinson_index(self, revenue_values: list, epsilon: float = 1.0) -> float:
        """Calculate Atkinson index from revenue values."""
        n = len(revenue_values)
        if n == 0:
            return 0.0
        total = sum(revenue_values)
        if total == 0:
            return 0.0
        mean = total / n
        try:
            positive_vals = [v for v in revenue_values if v > 0]
            if not positive_vals:
                return 0.0
            geo_mean = np.exp(sum(np.log(v) for v in positive_vals) / n)
            return 1 - (geo_mean / mean) ** (1 - epsilon) if epsilon != 1 else 0.0
        except (ValueError, ZeroDivisionError):
            return 0.0

    @staticmethod
    def sum_importance(scheduled: np.ndarray, revenue: Mapping[str, dict]) -> Mapping[str, float]:
        """Sum importance by RU from scheduled services."""
        scheduled_by_ru = {}
        for idx, (service_id, service_data) in enumerate(revenue.items()):
            if scheduled[idx]:
                ru = service_data['ru']
                importance = service_data.get('importance', 1.0)
                scheduled_by_ru[ru] = scheduled_by_ru.get(ru, 0) + importance
        return scheduled_by_ru

    @staticmethod
    def jains_fairness_index(
        scheduled: np.ndarray,
        capacities: Mapping[str, float],
        revenue: Mapping[str, dict]
    ) -> Tuple[float, Mapping[str, float]]:
        """
        Calculate Jain's fairness index for scheduled services.

        Args:
            scheduled: Boolean array indicating which services are scheduled.
            capacities: Capacity values for each RU.
            revenue: Revenue behavior mapping with 'ru' and 'importance' keys.

        Returns:
            Tuple of (fairness index [0,1], ratios dict)
        """
        scheduled_sum = FairnessMetrics.sum_importance(scheduled, revenue)

        if not scheduled_sum:
            return 1.0, {}

        ratios = {ru: scheduled_sum.get(ru, 0) / capacities.get(ru, 1) for ru in capacities}

        n = len(ratios)
        if n == 0:
            return 1.0, ratios

        sum_ratios = sum(ratios.values())
        sum_squares = sum(x ** 2 for x in ratios.values())

        if sum_squares == 0:
            return 0.0, ratios

        fairness = (sum_ratios ** 2) / (n * sum_squares)
        return fairness, scheduled_sum

    @staticmethod
    def gini_fairness_index(
        scheduled: np.ndarray,
        capacities: Mapping[str, float],
        revenue: Mapping[str, dict],
        alpha: float = 10.0
    ) -> Tuple[float, Mapping[str, float]]:
        """
        Calculate Gini-based fairness index for scheduled services.

        Args:
            scheduled: Boolean array indicating which services are scheduled.
            capacities: Capacity values for each RU.
            revenue: Revenue behavior mapping with 'ru' and 'importance' keys.
            alpha: Gini index parameter.

        Returns:
            Tuple of (fairness index [0,1], ratios dict)
        """
        scheduled_sum = FairnessMetrics.sum_importance(scheduled, revenue)

        for ru in scheduled_sum:
            scheduled_sum[ru] *= 1  # Could multiply by services_by_ru if needed

        if not scheduled_sum:
            raise ValueError("Scheduled resources cannot be empty.")

        if len(scheduled_sum) != len(capacities):
            raise ValueError("Scheduled resources and capacities must match.")

        ratios = {ru: scheduled_sum.get(ru, 0) / capacities.get(ru, 1) for ru in capacities}
        ratios = {ru: (ratio ** alpha) for ru, ratio in ratios.items()}

        values = list(ratios.values())
        n = len(values)
        total = sum(values)

        if total == 0:
            return 1.0, ratios

        sorted_values = sorted(values)
        cumulative = 0
        for i, value in enumerate(sorted_values, start=1):
            cumulative += i * value

        gini = (2 * cumulative) / (n * total) - (n + 1) / n
        fairness = 1 - gini
        return fairness, ratios

    @staticmethod
    def atkinson_fairness_index(
        scheduled: np.ndarray,
        capacities: Mapping[str, float],
        revenue: Mapping[str, dict],
        alpha: float = 10.0,
        epsilon: float = 2
    ) -> Tuple[float, Mapping[str, float]]:
        """
        Calculate Atkinson-based fairness index for scheduled services.

        Args:
            scheduled: Boolean array indicating which services are scheduled.
            capacities: Capacity values for each RU.
            revenue: Revenue behavior mapping with 'ru' and 'importance' keys.
            alpha: Weighting factor.
            epsilon: Inequality aversion parameter.

        Returns:
            Tuple of (fairness index [0,1], ratios dict)
        """
        scheduled_sum = FairnessMetrics.sum_importance(scheduled, revenue)

        if not scheduled_sum:
            return 1.0, {}

        ratios = {ru: scheduled_sum.get(ru, 0) / capacities.get(ru, 1) for ru in capacities}
        ratios = {ru: (ratio ** alpha) for ru, ratio in ratios.items()}

        values = list(ratios.values())
        n = len(values)
        if n == 0:
            return 1.0, ratios

        mean = sum(values) / n

        if mean == 0:
            return 1.0, ratios

        try:
            geo_mean = np.exp(sum(np.log(v) for v in values if v > 0) / n)
            atkinson_index = 1 - geo_mean / mean
        except (ValueError, ZeroDivisionError):
            atkinson_index = 0

        fairness = 1 - atkinson_index
        return fairness, ratios