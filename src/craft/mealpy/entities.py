"""Entities for Railway Scheduling Problem using MEALPY algorithms."""

import datetime
import numpy as np

from src.craft.entities import ConflictMatrix, Boundaries, Solution

from copy import deepcopy
from functools import cache
from math import e, cos, pi
from robin.supply.entities import TimeSlot, Line, Service, Supply
from robin.supply.generator.entities import ServiceScheduler
from typing import List, Mapping, Tuple, Union


class MealpyTimetabling:
    """
    Infrastructure Manager Revenue Maximization Problem Formulation.

    This class formulates and solves the revenue maximization problem for train scheduling.
    It maintains both the requested and updated schedules, computes operational times,
    enforces feasibility (via boundaries and conflict matrices) and evaluates the revenue.
    """

    def __init__(
        self,
        requested_services: List[Service],
        revenue_behavior: Mapping[str, Mapping[str, float]],
        line: Mapping[str, Tuple[float, float]],
        safe_headway: int = 10,
        max_stop_time: int = 10,
        fair_index: Union[None, str] = None,
        alpha: float = 1.0,
    ) -> None:
        """
        Initialize the MPTT instance.

        Args:
            requested_schedule: The requested schedule mapping.
            revenue_behavior: The revenue behavior parameters.
            line: Mapping of line station positions.
            safe_headway: The minimum safe headway time between trains.
            max_stop_time: Maximum allowed stop time.
            fair_index: The fairness index to use for equity considerations.
            alpha: Alpha parameter.
        """
        self.requested_services = requested_services
        self.reference_schedules = {service.id: service.schedule for service in self.requested_services}
        self.revenue = revenue_behavior
        self.line = line

        self.safe_headway = safe_headway
        self.im_mod_margin = 60
        self.max_stop_time = max_stop_time
        self.fair_index = fair_index
        self.alpha = alpha

        self.n_services = len(self.requested_services)
        self.operational_times = self.get_operational_times()
        self.services_by_ru = self.get_n_services_by_ru()
        self.capacities = self.get_capacities()

        # Build reference solution and service indexer (for real variables)
        reference_solution = []
        service_indexer = []
        for service_id, service_schedule in self.reference_schedules.values():
            for _, departure_time in service_schedule[:-1]:
                reference_solution.append(departure_time)
                service_indexer.append(service_id)
        self.reference_solution = tuple(reference_solution)
        self.service_indexer = tuple(service_indexer)
        self.updated_schedule = deepcopy(self.reference_schedules)
        self.boundaries = self._calculate_boundaries()
        self.conflict_matrix = self._get_conflict_matrix()
        self.best_revenue = -np.inf
        self.best_solution = None
        self.feasible_schedules = []
        self.dt_indexer = self.get_departure_time_indexer()
        self.indexer = {sch: idx for idx, sch in enumerate(self.requested_schedule)}
        self.rev_indexer = {idx: sch for idx, sch in enumerate(self.requested_schedule)}
        self.requested_times = self.get_real_vars()
        self.scheduled_trains = np.zeros(self.n_services, dtype=bool)

    # === Public Interface Methods ===

    def update_supply(self, path: str, solution: Solution) -> List[Service]:
        """
        Update the supply based on the provided solution.

        Args:
            path: Path to the YAML file containing the supply.
            solution: The solution containing discrete scheduling decisions.

        Returns:
            A list of updated Service objects.
        """
        self.update_schedule(solution)
        services = []
        supply = Supply.from_yaml(path=path)
        scheduled_services = solution.discrete

        if len(scheduled_services) != len(supply.services):
            raise AssertionError("Scheduled services and services in supply do not match")

        for S_i, service in zip(scheduled_services, supply.services):
            if not S_i:
                continue

            service_schedule = self.updated_schedule[service.id]
            # Convert schedule times to floats
            timetable = {sta: tuple(map(float, times)) for sta, times in service_schedule.items()}
            departure_time = list(timetable.values())[0][1]
            # Calculate timetable relative to the departure time
            relative_timetable = {
                sta: tuple(float(t) - departure_time for t in times)
                for sta, times in service_schedule.items()
            }
            updated_line_id = str(hash(str(list(relative_timetable.values()))))
            updated_line = Line(updated_line_id, service.line.name, service.line.corridor, relative_timetable)
            date = service.date
            start_time = datetime.timedelta(minutes=float(departure_time))
            time_slot_id = f"{start_time.seconds}"
            updated_time_slot = TimeSlot(time_slot_id, start_time, start_time + datetime.timedelta(minutes=10))
            updated_service = Service(
                id_=service.id,
                date=date,
                line=updated_line,
                tsp=service.tsp,
                time_slot=updated_time_slot,
                rolling_stock=service.rolling_stock,
                prices=service.prices
            )
            services.append(updated_service)
        return services

    def update_schedule(self, solution: np.array) -> None:
        """
        Update the schedule using the provided solution.

        Args:
            solution: Array of departure times (real variables) for scheduling.
        """
        departure_times = solution if solution.any() else self.get_real_vars()
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
                    arrival_time = departure_times[dt_idx - 1] + self.operational_times[service][ot_idx]
                    departure_time = arrival_time
                else:
                    arrival_time = departure_times[dt_idx - 1] + self.operational_times[service][ot_idx]
                    departure_time = departure_times[dt_idx]
                    ot_idx += 2
                    dt_idx += 1

                self.updated_schedule[service][stop][0] = arrival_time
                self.updated_schedule[service][stop][1] = departure_time

        # Recalculate boundaries and conflict matrix after updating times
        self.boundaries = self._calculate_boundaries()
        self.conflict_matrix = self._get_conflict_matrix()

    def update_feasible_schedules(self, solution: List[float]) -> None:
        """
        Update feasible schedules based on the provided solution.

        Args:
            solution: List of departure times (real variables).
        """
        self.update_schedule(solution)
        # Generate all possible binary schedules (truth table) for n_services
        train_combinations = self.truth_table(dim=self.n_services)
        self.feasible_schedules = [S_i for S_i in train_combinations if self._departure_time_feasibility(S_i)]

    def objective_function(self, solution: List[float]) -> float:
        """
        Compute the fitness (objective value) for the provided solution.
        If 'equity' is True, the revenue is multiplied by Jain's fairness index.

        Args:
            solution: List of departure times.

        Returns:
            Fitness value (float).
        """
        solution_arr = np.array(solution, dtype=np.int32)
        self.update_schedule(solution_arr)
        schedule = self.get_heuristic_schedule_old()
        fairness = 1.0

        revenue = self.get_revenue(Solution(real=solution, discrete=schedule))
        return revenue * fairness

    def get_revenue(self, solution: Solution) -> float:
        """
        Compute the total revenue for the given solution.

        Args:
            solution: A Solution object containing real and discrete scheduling decisions.

        Returns:
            Total revenue (float).
        """
        S_i = solution.discrete
        im_revenue = 0.0
        for idx, service in enumerate(self.requested_schedule):
            if S_i[idx] and self.service_is_feasible(service):
                im_revenue += self.get_service_revenue(service)

        if im_revenue > self.best_revenue:
            self.best_revenue = im_revenue
            self.best_solution = solution

        return im_revenue

    def is_feasible(
            self, timetable: Solution, scheduling: np.array, update_schedule: bool = True
    ) -> bool:
        """
        Check if the provided solution is feasible.

        Args:
            timetable: The solution obtained from the optimization algorithm.
            scheduling: Boolean array representing discrete scheduling decisions.
            update_schedule: Whether to update the schedule with the provided timetable.

        Returns:
            True if the solution is feasible, False otherwise.
        """
        if update_schedule:
            self.update_schedule(timetable)

        if not self._feasible_boundaries(timetable):
            return False

        dt_feasible = self._departure_time_feasibility(scheduling)
        tt_feasible = self._travel_times_feasibility(scheduling)
        return dt_feasible and tt_feasible

    def get_best_schedule(self, solution: List[float]) -> np.array:
        """
        Determine the best feasible schedule based on revenue maximization.

        Args:
            solution: List of departure times from the optimization algorithm.

        Returns:
            Best feasible schedule as a numpy array.
        """
        self.update_feasible_schedules(solution)
        best_schedule = None
        best_revenue = -np.inf
        for fs in self.feasible_schedules:
            revenue = self.get_revenue(Solution(real=solution, discrete=fs))
            if revenue > best_revenue:
                best_revenue = revenue
                best_schedule = fs
        return np.array(best_schedule) if best_schedule is not None else np.array([])

    def get_heuristic_schedule(self) -> np.array:
        """
        Compute the best schedule using an older (conflict‐avoiding sequential) heuristic.

        Returns:
            Final schedule as a boolean numpy array.
        """
        default_planner = np.array([not cm.any() for cm in self.conflict_matrix], dtype=bool)
        conflicts = {sch for sch in self.updated_schedule if not default_planner[self.indexer[sch]]}
        conflicts_revenue = {sc: self.get_service_revenue(sc) for sc in conflicts}
        # Sort by revenue (lowest first)
        conflicts_revenue = dict(sorted(conflicts_revenue.items(), key=lambda item: item[1]))

        while conflicts_revenue:
            # Select the service with the highest revenue among the conflicts.
            s = next(reversed(conflicts_revenue))
            default_planner[self.indexer[s]] = True
            # Eliminar s de conflicts_revenue para evitar bucle infinito.
            conflicts_revenue.pop(s, None)
            conflicts_with_s = {self.rev_indexer[idx] for idx in np.where(self.conflict_matrix[self.indexer[s]])[0]}
            conflicts_revenue = {k: v for k, v in conflicts_revenue.items() if k not in conflicts_with_s}
        return default_planner

    def get_operational_times(self) -> Mapping[str, List[float]]:
        """
        Compute operational times for each service based on the requested schedule.

        Returns:
            A mapping from service to a list of operational times.
        """
        operational_times = {}
        for service, stops in self.requested_schedule.items():
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

    def get_real_vars(self) -> List[int]:
        """
        Extract the real variables (departure times) from the requested schedule.

        Returns:
            List of departure times.
        """
        real_vars = []
        for service, stops in self.requested_schedule.items():
            stop_keys = list(stops.keys())
            for i in range(len(stop_keys) - 1):
                real_vars.append(stops[stop_keys[i]][1])
        return real_vars

    def get_service_revenue(self, service: str) -> float:
        """
        Compute the revenue for a given service based on its updated schedule.

        Args:
            service: Service identifier.

        Returns:
            Revenue value (float) for the service.
        """
        k = self.revenue[service]["k"]
        departure_station = list(self.requested_schedule[service].keys())[0]
        departure_time_delta = abs(
            self.updated_schedule[service][departure_station][1] -
            self.requested_schedule[service][departure_station][1]
        )
        tt_penalties = []
        stop_keys = list(self.requested_schedule[service].keys())
        for j, stop in enumerate(stop_keys):
            if j == 0 or j == len(stop_keys) - 1:
                continue
            penalty_val = self.penalty_function(
                abs(self.updated_schedule[service][stop][1] - self.requested_schedule[service][stop][
                    1]) / self.im_mod_margin,
                k,
            )
            tt_penalties.append(penalty_val * self.revenue[service]["tt_max_penalty"])
        dt_penalty = self.penalty_function(departure_time_delta / self.im_mod_margin, k) * self.revenue[service][
            "dt_max_penalty"]
        return self.revenue[service]["canon"] - dt_penalty - np.sum(tt_penalties)

    def service_is_feasible(self, service: str) -> bool:
        """
        Check if the updated schedule for a service is feasible relative to its requested schedule.

        Args:
            service: Service identifier.

        Returns:
            True if the service schedule is feasible, False otherwise.
        """
        original_times = list(self.requested_schedule[service].values())
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

    def get_n_services_by_ru(self) -> Mapping[str, int]:
        """
        Count the number of services per RU based on revenue behavior.

        Returns:
            A mapping from RU to service count.
        """
        services_by_ru = {}
        for service, data in self.revenue.items():
            ru = data["ru"]
            services_by_ru[ru] = services_by_ru.get(ru, 0) + 1
        return services_by_ru

    def get_capacities(self) -> Mapping[str, float]:
        """
        Calculate capacities for each RU as a percentage of total services.

        Returns:
            A mapping from RU to capacity percentage.
        """
        return {ru: (count / self.n_services) * 100 for ru, count in self.services_by_ru.items()}

    def get_departure_time_indexer(self) -> Mapping[int, str]:
        """
        Build an index mapping where keys are departure time indices and values are the corresponding service IDs.

        Returns:
            A mapping from integer index to service identifier.
        """
        dt_indexer = {}
        i = 0
        for service, stops in self.requested_schedule.items():
            # Each service provides (number of stops - 1) departure times.
            for _ in range(len(stops) - 1):
                dt_indexer[i] = service
                i += 1
        return dt_indexer

    def _calculate_boundaries(self) -> Boundaries:
        """
        Calculate boundaries for the departure times of each service.

        Returns:
            A Boundaries object containing the real (and empty discrete) boundaries.
        """
        boundaries = []
        for service, stops in self.requested_schedule.items():
            stop_keys = list(stops.keys())
            ot_idx = 0
            for i in range(len(stop_keys) - 1):
                if i == 0:
                    lower_bound = stops[stop_keys[i]][1] - self.im_mod_margin
                    upper_bound = stops[stop_keys[i]][1] + self.im_mod_margin
                else:
                    travel_time = self.operational_times[service][ot_idx]
                    stop_time = self.operational_times[service][ot_idx + 1]
                    ot_idx += 2
                    lower_bound = self.updated_schedule[service][stop_keys[i - 1]][1] + travel_time + stop_time
                    max_dt_original = stops[stop_keys[i]][1] + self.max_stop_time
                    max_dt_updated = lower_bound + (self.max_stop_time - stop_time)
                    upper_bound = min(max_dt_original, max_dt_updated)
                boundaries.append([lower_bound, upper_bound])
        return Boundaries(real=boundaries, discrete=[])

    def _departure_time_feasibility(self, S_i: np.array) -> bool:
        """
        Check whether the departure times in the solution are conflict‐free.

        Args:
            S_i: A boolean array of scheduling decisions.

        Returns:
            True if no conflicts exist; otherwise, False.
        """
        S_i_bool = np.array(S_i, dtype=bool)
        return not np.any((S_i_bool * self.conflict_matrix)[S_i_bool])

    def _feasible_boundaries(self, solution: Solution) -> bool:
        """
        Check that each real variable in the solution lies within its corresponding boundary.

        Args:
            solution: A Solution object containing real departure times.

        Returns:
            True if all values are within bounds; otherwise, False.
        """
        return all(
            self.boundaries.real[i][0] <= rv <= self.boundaries.real[i][1]
            for i, rv in enumerate(solution.real)
        )

    def _get_conflict_matrix(self) -> np.array:
        """
        Compute the conflict matrix among services based on the updated schedule.

        Returns:
            A boolean numpy array where each entry [i, j] indicates whether service i and service j conflict.
        """
        conflict_matrix = ConflictMatrix(services=self.updated_services)

        for i, service in enumerate(self.updated_services):
            service_scheduler = ServiceScheduler(services=self.updated_services[i+1:])
            conflicts_ids = service_scheduler.find_conflicts(service)
            for conflict_id in conflicts_ids:
                conflict_matrix.set(service.id, conflict_id, True)
        return conflict_matrix

    def _travel_times_feasibility(self, S_i: np.array) -> bool:
        """
        Check whether the travel times in the updated schedule are feasible.

        Args:
            S_i: A boolean array of scheduling decisions.

        Returns:
            True if travel times are feasible; otherwise, False.
        """
        for i, service in enumerate(self.requested_schedule):
            if not S_i[i]:
                continue
            if not self.service_is_feasible(service):
                return False
        return True

    @staticmethod
    def penalty_function(x: float, k: int) -> float:
        """
        Compute the penalty based on a normalized deviation.

        Args:
            x: Normalized deviation.
            k: Scaling factor.

        Returns:
            Penalty value (float).
        """
        return 1 - e ** (-k * x ** 2) * (0.5 * cos(pi * x) + 0.5)

    @cache
    def truth_table(self, dim: int) -> List[List[int]]:
        """
        Generate a truth table (all binary combinations) for the given dimension.

        Args:
            dim: Dimension of the truth table.

        Returns:
            A list of binary combinations.
        """
        if dim < 1:
            return [[]]
        sub_tt = self.truth_table(dim - 1)
        return [row + [val] for row in sub_tt for val in [0, 1]]
