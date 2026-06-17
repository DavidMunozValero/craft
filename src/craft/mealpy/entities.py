"""Railway Scheduling Problem using MEALPY algorithms."""

import datetime
import numpy as np

from ..entities import Boundaries, Solution, ConflictMatrix

from copy import deepcopy
from robin.supply.entities import TimeSlot, Line, Service, Supply
from robin.supply.generator.entities import ServiceScheduler

from .revenue import RevenueCalculator
from .scheduling import ScheduleManager
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
        safe_headway: int = 10,
        max_stop_time: int = 10,
        fair_index: Union[None, str] = None,
        alpha: float = 1.0,
    ) -> None:
        self.requested_services = requested_services
        self.updated_services = deepcopy(self.requested_services)
        self.reference_schedules = self._build_reference_schedules()
        self.revenue = revenue_behavior

        self.safe_headway = safe_headway
        self.im_mod_margin = 60
        self.max_stop_time = max_stop_time
        self.fair_index = fair_index
        self.alpha = alpha

        self.n_services = len(self.requested_services)
        
        self.schedule_manager = ScheduleManager(
            self.reference_schedules,
            {}
        )
        self.operational_times = self.schedule_manager.compute_operational_times()
        self.schedule_manager.operational_times = self.operational_times
        
        self.services_by_ru = self._get_services_by_ru()
        self.capacities = self._get_capacities()
        self.revenue_calculator = RevenueCalculator(
            self.revenue,
            self.reference_schedules,
            self.schedule_manager.updated_schedule,
            self.im_mod_margin
        )

        self.reference_solution, self.service_indexer = self._build_reference_solution()
        self.boundaries = self._calculate_boundaries()
        self.conflict_matrix = self._get_conflict_matrix()
        self.best_revenue = -np.inf
        self.best_solution = None
        self.dt_indexer = self.schedule_manager.get_departure_time_indexer()
        self.indexer = {sch: idx for idx, sch in enumerate(self.reference_schedules)}
        self.rev_indexer = {idx: sch for idx, sch in enumerate(self.reference_schedules)}
        self.requested_times = self._get_real_vars()

    def _build_reference_schedules(self) -> Mapping[str, dict]:
        """Build reference schedules from requested services."""
        return {
            service.id: {
                station: list(map(lambda x: x.total_seconds() // 60, service.schedule[station]))
                for station in service.schedule
            }
            for service in self.requested_services
        }

    def _compute_operational_times(self) -> Mapping[str, List[float]]:
        """Compute operational times for each service."""
        return self.operational_times

    def _get_services_by_ru(self) -> Mapping[str, int]:
        """Count services per RU."""
        services_by_ru = {}
        for service, data in self.revenue.items():
            ru = data["ru"]
            services_by_ru[ru] = services_by_ru.get(ru, 0) + 1
        return services_by_ru

    def _get_capacities(self) -> Mapping[str, float]:
        """Calculate capacities as percentage of total services."""
        return {ru: (count / self.n_services) * 100 for ru, count in self.services_by_ru.items()}

    def _build_reference_solution(self) -> Tuple[tuple, tuple]:
        """Build reference solution and service indexer."""
        reference_solution = []
        service_indexer = []
        for service_id, service_schedule in self.reference_schedules.items():
            for _, departure_time in tuple(service_schedule.values())[:-1]:
                reference_solution.append(departure_time)
                service_indexer.append(service_id)
        return tuple(reference_solution), tuple(service_indexer)

    def _get_real_vars(self) -> List[int]:
        """Extract real variables from reference schedule."""
        return self.schedule_manager._get_real_vars()

    def update_supply(self, path: str, solution: Solution) -> List[Service]:
        """Update supply based on solution."""
        self.schedule_manager.update_from_solution(solution.real)

        services = []
        supply = Supply.from_yaml(path=path)
        scheduled_services = solution.discrete

        if len(scheduled_services) != len(supply.services):
            raise AssertionError("Scheduled services and services in supply do not match")

        for S_i, service in zip(scheduled_services, supply.services):
            if not S_i:
                continue

            service_schedule = self.schedule_manager.updated_schedule[service.id]
            timetable = {sta: tuple(map(float, times)) for sta, times in service_schedule.items()}
            departure_time = list(timetable.values())[0][1]

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

    def update_schedule(self, solution: np.ndarray) -> None:
        """Update schedule from solution."""
        self.schedule_manager.update_from_solution(solution)
        self.boundaries = self._calculate_boundaries()
        self.conflict_matrix = self._get_conflict_matrix()

    def objective_function(self, solution: List[float]) -> float:
        """Compute fitness for the optimization algorithm."""
        solution_arr = np.array(solution, dtype=np.int32)
        self.schedule_manager.update_from_solution(solution_arr)

        self.revenue_calculator.updated_schedule = self.schedule_manager.updated_schedule
        self.revenue_calculator.recompute_all_revenues()
        schedule = self.get_heuristic_schedule()

        revenue = self.get_revenue(Solution(real=solution, discrete=schedule))
        return revenue

    def get_revenue(self, solution: Solution) -> float:
        """Compute total revenue for solution."""
        S_i = solution.discrete
        im_revenue = 0.0

        service_ids = list(self.reference_schedules.keys())
        for idx, service in enumerate(service_ids):
            if S_i[idx] and self.schedule_manager.is_service_feasible(service):
                im_revenue += self.revenue_calculator.get_service_revenue(service)

        if im_revenue > self.best_revenue:
            self.best_revenue = im_revenue
            self.best_solution = solution

        return im_revenue

    def is_feasible(self, timetable: Solution, scheduling: np.ndarray, update_schedule: bool = True) -> bool:
        """Check if solution is feasible."""
        if update_schedule:
            self.schedule_manager.update_from_solution(timetable.real)
            self.boundaries = self._calculate_boundaries()
            self.conflict_matrix = self._get_conflict_matrix()

        if not self._feasible_boundaries(timetable):
            return False

        dt_feasible = self._departure_time_feasibility(scheduling)
        tt_feasible = self._travel_times_feasibility(scheduling)
        return dt_feasible and tt_feasible

    def get_heuristic_schedule(self) -> np.ndarray:
        """Compute schedule using conflict-avoiding heuristic."""
        default_planner = np.array([not cm.any() for cm in self.conflict_matrix.matrix], dtype=bool)
        conflicts = {sch for sch in self.schedule_manager.updated_schedule if not default_planner[self.indexer[sch]]}

        conflicts_revenue = {
            sc: self.revenue_calculator.get_service_revenue(sc)
            for sc in conflicts
        }
        conflicts_revenue = dict(sorted(conflicts_revenue.items(), key=lambda item: item[1]))

        while conflicts_revenue:
            s = next(reversed(conflicts_revenue))
            default_planner[self.indexer[s]] = True
            conflicts_revenue.pop(s, None)

            conflicts_with_s = {
                self.rev_indexer[idx]
                for idx in np.where(self.conflict_matrix.matrix[self.indexer[s]])[0]
            }
            conflicts_revenue = {
                k: v for k, v in conflicts_revenue.items()
                if k not in conflicts_with_s
            }
        return default_planner

    def _calculate_boundaries(self) -> Boundaries:
        """Calculate boundaries for departure times."""
        boundaries = []
        prev_lower_bound = None

        for service, stops in self.reference_schedules.items():
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

                    lower_bound = prev_lower_bound + travel_time + stop_time
                    max_dt_original = stops[stop_keys[i]][1] + self.max_stop_time
                    max_dt_updated = lower_bound + (self.max_stop_time - stop_time)
                    upper_bound = max(max_dt_original, max_dt_updated)

                boundaries.append([lower_bound, upper_bound])
                prev_lower_bound = lower_bound

        return Boundaries(real=boundaries, discrete=[])

    def _departure_time_feasibility(self, S_i: np.ndarray) -> bool:
        """Check departure times are conflict-free."""
        S_i_bool = np.array(S_i, dtype=bool)
        return not np.any((S_i_bool * self.conflict_matrix.matrix)[S_i_bool])

    def _feasible_boundaries(self, solution: Solution) -> bool:
        """Check all real variables are within boundaries."""
        return all(
            self.boundaries.real[i][0] <= rv <= self.boundaries.real[i][1]
            for i, rv in enumerate(solution.real)
        )

    def _get_conflict_matrix(self) -> ConflictMatrix:
        """Compute conflict matrix among services."""
        conflict_matrix = ConflictMatrix(services=self.updated_services)

        service_scheduler = ServiceScheduler(services=self.updated_services)
        for service in self.updated_services:
            conflicts_ids = service_scheduler.find_conflicts(
                new_service=service,
                safety_gap=self.safe_headway
            )
            for conflict_id in conflicts_ids:
                conflict_matrix.set(service.id, conflict_id, True)

        return conflict_matrix

    def _travel_times_feasibility(self, S_i: np.ndarray) -> bool:
        """Check travel times are feasible."""
        for i, service in enumerate(self.reference_schedules):
            if not S_i[i]:
                continue
            if not self.schedule_manager.is_service_feasible(service):
                return False
        return True