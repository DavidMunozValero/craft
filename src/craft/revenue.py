"""Revenue modeling for railway services.

This module bundles the revenue-related pieces shared by the optimization
backends: the penalty function used to discourage timetable deviations, the
``RevenueSimulator`` that synthesizes a revenue behavior for each service of
a supply, and the ``RevenueCalculator`` that evaluates the revenue of a
candidate timetable against that behavior.
"""

from math import cos, e, pi
from typing import Any, Callable, Mapping

import numpy as np
from geopy.distance import geodesic
from scipy.stats import loguniform

from robin.supply.entities import Supply


def penalty_function(x: float, k: float) -> float:
    """Compute the penalty derived from a normalized timetable deviation.

    The penalty grows with the normalized deviation ``x`` (deviation divided
    by the admissible modification margin) and is shaped by the scaling
    factor ``k``. It is bounded in ``[0, 1]`` and equals ``0`` when there is
    no deviation.

    Args:
        x: Normalized deviation.
        k: Scaling factor.

    Returns:
        Penalty value in ``[0, 1]``.
    """
    return float(1 - e ** (-k * x**2) * (0.5 * cos(pi * x) + 0.5))


class RevenueSimulator:
    """Synthesize a revenue behavior for every service of a supply.

    The behavior of a service is described by a canon (base revenue), a
    sensitivity factor ``k``, the maximum penalties for departure-time and
    travel-time deviations and a normalized importance weight within its RU
    (Railway Undertaking / TSP) group.
    """

    def __init__(self, supply: Supply) -> None:
        self.supply = supply

    def simulate_revenue(
        self, alpha: float = 2 / 3
    ) -> Mapping[str, Mapping[str, float]]:
        """Calculate revenue behavior parameters for each service in the supply.

        For every service in the supply, this function computes:
          - canon: The base revenue increased by a randomly selected bias factor.
          - k: A random scaling factor drawn from a log-uniform distribution.
          - dt_max_penalty: A penalty value derived from the canon.
          - tt_max_penalty: A penalty value distributed across the service's stations.
          - importance: A normalized random weight assigned to the service within its RU group.

        The services are first grouped by their associated RU (Transport Service Provider).
        Then, for each group, a set of random values is generated and normalized so that the
        sum of importance values in each group is 1. Finally, these importance values are assigned
        to the corresponding services.

        Args:
            alpha: Capacity share contributing to the canon (defaults to 2/3).

        Returns:
            A dictionary mapping each service's ID to a dictionary of computed revenue parameters.
        """
        revenue: dict[str, dict[str, float]] = {}
        tsp_k: dict[str, float] = {}
        tsp_importance: dict[str, Any] = {}

        tsp_groups: dict[str, list] = {}
        for service in self.supply.services:
            if service.tsp.id not in tsp_groups:
                tsp_groups[service.tsp.id] = []
            tsp_groups[service.tsp.id].append(service)

        for tsp_id, services in tsp_groups.items():
            n_services_tsp = len(services)
            random_importances = np.random.random(n_services_tsp)
            normalized = random_importances / random_importances.sum()
            tsp_importance[tsp_id] = iter(normalized)

        for service in self.supply.services:
            sta_coords = [sta.coordinates for sta in service.line.stations]
            distances = [
                geodesic(sta_coords[i], sta_coords[i + 1]).km
                for i in range(len(sta_coords) - 1)
            ]
            distance_factor = 7 * sum(distances)
            service_capacity = service.rolling_stock.total_capacity
            capacity_factor = (alpha * service_capacity) / 100 * 1.67
            stations_factor = 18 + (len(sta_coords) - 2) * 65 + 165
            total_canon = (distance_factor + capacity_factor + stations_factor) / 100
            max_penalty = total_canon * 0.3
            dt_penalty = np.round(max_penalty * 0.35, 2)
            tt_penalty = np.round((max_penalty - dt_penalty) / (len(sta_coords) - 1), 2)
            k = tsp_k.get(service.tsp.id)
            if k is None:
                k = np.round(loguniform.rvs(0.01, 100, 1), 2)
                tsp_k[service.tsp.id] = k

            importance = next(tsp_importance[service.tsp.id])

            revenue[service.id] = {
                "canon": total_canon,
                "ru": service.tsp.id,
                "k": k,
                "dt_max_penalty": dt_penalty,
                "tt_max_penalty": tt_penalty,
                "importance": importance,
            }

        return revenue


class RevenueCalculator:
    """Evaluate the revenue of a candidate timetable.

    Given a revenue behavior (as produced by :class:`RevenueSimulator`), the
    reference schedules and an updated schedule, it computes the revenue of
    each service by discounting the canon with the departure-time and
    travel-time penalties.
    """

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
        self.service_revenues: dict[str, dict[str, float]] = {}
        self._compute_all_revenues()

    def _compute_all_revenues(self) -> None:
        """Pre-compute revenue for all services."""
        for service in self.reference_schedules.keys():
            rev = self.get_service_revenue(service)
            self.service_revenues[service] = {
                "revenue": rev,
                "canon": self.revenue[service]["canon"],
                "importance": self.revenue[service]["importance"],
                "ru": self.revenue[service]["ru"],
            }

    def get_service_revenue(self, service: str) -> float:
        """Compute revenue for a specific service."""
        k = self.revenue[service]["k"]
        departure_station = list(self.reference_schedules[service].keys())[0]

        departure_time_delta = abs(
            self.updated_schedule[service][departure_station][1]
            - self.reference_schedules[service][departure_station][1]
        )

        tt_penalties = []
        stop_keys = list(self.reference_schedules[service].keys())
        for j, stop in enumerate(stop_keys):
            if j == 0 or j == len(stop_keys) - 1:
                continue
            penalty_val = penalty_function(
                abs(
                    self.updated_schedule[service][stop][1]
                    - self.reference_schedules[service][stop][1]
                )
                / self.im_mod_margin,
                k,
            )
            tt_penalties.append(penalty_val * self.revenue[service]["tt_max_penalty"])

        dt_penalty = (
            penalty_function(departure_time_delta / self.im_mod_margin, k)
            * self.revenue[service]["dt_max_penalty"]
        )

        return float(self.revenue[service]["canon"] - dt_penalty - np.sum(tt_penalties))

    def recompute_all_revenues(self) -> None:
        """Recompute revenue for all services after schedule update."""
        for service in self.reference_schedules.keys():
            rev = self.get_service_revenue(service)
            self.service_revenues[service]["revenue"] = rev

    def compute_total_revenue(
        self,
        service_ids: list,
        scheduled_mask: np.ndarray,
        feasibility_checker: Callable,
    ) -> float:
        """Compute total revenue for scheduled services."""
        total = 0.0
        for idx, service_id in enumerate(service_ids):
            if scheduled_mask[idx] and feasibility_checker(service_id):
                total += self.get_service_revenue(service_id)
        return total
