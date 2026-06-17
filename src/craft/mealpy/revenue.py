"""Revenue calculation for railway services."""

import numpy as np

from .utils import penalty_function
from typing import Mapping


class RevenueCalculator:
    """Handles revenue calculations for services."""

    def __init__(
        self,
        revenue_behavior: Mapping[str, Mapping[str, float]],
        reference_schedules: Mapping[str, dict],
        updated_schedule: dict,
        im_mod_margin: int = 60,
    ) -> None:
        self.revenue = revenue_behavior
        self.reference_schedules = reference_schedules
        self.updated_schedule = updated_schedule
        self.im_mod_margin = im_mod_margin
        self.service_revenues = {}
        self._compute_all_revenues()

    def _compute_all_revenues(self) -> None:
        """Pre-compute revenue for all services."""
        for service in self.reference_schedules.keys():
            rev = self.get_service_revenue(service)
            self.service_revenues[service] = {
                'revenue': rev,
                'canon': self.revenue[service]['canon'],
                'importance': self.revenue[service]['importance'],
                'ru': self.revenue[service]['ru']
            }

    def get_service_revenue(self, service: str) -> float:
        """Compute revenue for a specific service."""
        k = self.revenue[service]["k"]
        departure_station = list(self.reference_schedules[service].keys())[0]

        departure_time_delta = abs(
            self.updated_schedule[service][departure_station][1] -
            self.reference_schedules[service][departure_station][1]
        )

        tt_penalties = []
        stop_keys = list(self.reference_schedules[service].keys())
        for j, stop in enumerate(stop_keys):
            if j == 0 or j == len(stop_keys) - 1:
                continue
            penalty_val = penalty_function(
                abs(self.updated_schedule[service][stop][1] -
                    self.reference_schedules[service][stop][1]) / self.im_mod_margin,
                k,
            )
            tt_penalties.append(penalty_val * self.revenue[service]["tt_max_penalty"])

        dt_penalty = penalty_function(
            departure_time_delta / self.im_mod_margin, k
        ) * self.revenue[service]["dt_max_penalty"]

        return self.revenue[service]["canon"] - dt_penalty - np.sum(tt_penalties)

    def recompute_all_revenues(self) -> None:
        """Recompute revenue for all services after schedule update."""
        for service in self.reference_schedules.keys():
            rev = self.get_service_revenue(service)
            self.service_revenues[service]['revenue'] = rev

    def compute_total_revenue(
        self,
        service_ids: list,
        scheduled_mask: np.ndarray,
        feasibility_checker: callable
    ) -> float:
        """Compute total revenue for scheduled services."""
        total = 0.0
        for idx, service_id in enumerate(service_ids):
            if scheduled_mask[idx] and feasibility_checker(service_id):
                total += self.get_service_revenue(service_id)
        return total