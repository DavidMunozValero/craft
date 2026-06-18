"""Schedule management for railway services."""

from copy import deepcopy
from pathlib import Path
from typing import List, Mapping, Union

import numpy as np

from robin.supply.entities import Supply


class ScheduleManager:
    """Manages service schedules and operational times."""

    def __init__(
        self,
        reference_schedules: Mapping[str, dict],
        operational_times: Mapping[str, List[float]],
    ) -> None:
        self.reference_schedules = reference_schedules
        self.operational_times = operational_times
        self.updated_schedule = deepcopy(reference_schedules)

    def update_from_solution(self, solution: np.ndarray) -> None:
        """Update schedule based on optimization solution."""
        departure_times = solution if solution.any() else self._get_real_vars()
        dt_idx = 0

        for service in self.updated_schedule:
            ot_idx = 0
            stops = list(self.updated_schedule[service].keys())
            for j, stop in enumerate(stops):
                if j == 0:
                    departure_time = departure_times[dt_idx]
                    arrival_time = departure_time
                    dt_idx += 1
                elif j == len(stops) - 1:
                    arrival_time = (
                        departure_times[dt_idx - 1]
                        + self.operational_times[service][ot_idx]
                    )
                    departure_time = arrival_time
                else:
                    arrival_time = (
                        departure_times[dt_idx - 1]
                        + self.operational_times[service][ot_idx]
                    )
                    departure_time = departure_times[dt_idx]
                    ot_idx += 2
                    dt_idx += 1

                self.updated_schedule[service][stop][0] = arrival_time
                self.updated_schedule[service][stop][1] = departure_time

    def _get_real_vars(self) -> List[int]:
        """Extract real variables (departure times) from reference schedule."""
        real_vars = []
        for service, stops in self.reference_schedules.items():
            stop_keys = list(stops.keys())
            for i in range(len(stop_keys) - 1):
                real_vars.append(stops[stop_keys[i]][1])
        return real_vars

    def is_service_feasible(self, service: str) -> bool:
        """Check if updated schedule maintains travel times."""
        original_times = list(self.reference_schedules[service].values())
        updated_times = list(self.updated_schedule[service].values())

        for j in range(len(original_times) - 1):
            original_tt = original_times[j + 1][0] - original_times[j][1]
            updated_tt = updated_times[j + 1][0] - updated_times[j][1]
            if updated_tt < original_tt:
                return False

            if j > 0:
                original_st = original_times[j][1] - original_times[j][0]
                updated_st = updated_times[j][1] - updated_times[j][0]
                if updated_st < original_st:
                    return False
        return True

    def compute_operational_times(self) -> Mapping[str, List[float]]:
        """Compute operational times for each service."""
        operational_times = {}
        for service, stops in self.reference_schedules.items():
            stop_keys = list(stops.keys())
            times = []
            for i in range(len(stop_keys) - 1):
                origin = stop_keys[i]
                destination = stop_keys[i + 1]
                travel_time = stops[destination][0] - stops[origin][1]
                if i == 0:
                    times.append(travel_time)
                else:
                    stop_time = stops[origin][1] - stops[origin][0]
                    times.extend([stop_time, travel_time])
            operational_times[service] = times
        return operational_times

    def get_departure_time_indexer(self) -> Mapping[int, str]:
        """Build index mapping departure time index to service ID."""
        dt_indexer = {}
        i = 0
        for service, stops in self.reference_schedules.items():
            for _ in range(len(stops) - 1):
                dt_indexer[i] = service
                i += 1
        return dt_indexer


def get_schedule_from_supply(
    path: Union[Path, None] = None,
    supply: Union[Supply, None] = None,
) -> Mapping[str, Mapping[str, List[int]]]:
    """Build a per-service schedule mapping from a robin ``Supply``.

    The schedule is expressed in minutes since midnight, with each stop
    mapped to ``[arrival_time, departure_time]``. Either ``path`` (to a
    supply YAML file) or a ready ``supply`` object must be provided.
    """
    if not supply:
        supply = Supply.from_yaml(path=path)
    requested_schedule: dict[str, dict[str, list[int]]] = {}
    for service in supply.services:
        requested_schedule[service.id] = {}
        time = service.time_slot.start
        delta = time.total_seconds() // 60
        for stop in service.line.timetable:
            arrival_time = delta + int(service.line.timetable[stop][0])
            departure_time = delta + int(service.line.timetable[stop][1])
            requested_schedule[service.id][stop] = [arrival_time, departure_time]

    return requested_schedule
