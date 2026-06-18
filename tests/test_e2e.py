"""End-to-end smoke tests for the optimization backends.

Runs a tiny mealpy GA and a tiny GSA on the 8-service test supply with a
fixed seed, verifying that the full pipeline (revenue -> timetabling ->
optimize -> schedule) produces sensible results.
"""

import numpy as np

from craft import Solution
from craft.gsa import Boundaries, GSA
from craft.mealpy import MealpyTimetabling


class TestMealpyE2E:
    def test_ga_smoke(self, supply, revenue_behavior):
        from mealpy import FloatVar, GA

        tt = MealpyTimetabling(
            requested_services=supply.services,
            revenue_behavior=revenue_behavior,
            safe_headway=10,
            max_stop_time=10,
        )
        bounds = [FloatVar(lb=lb, ub=ub) for lb, ub in tt.boundaries.real]
        problem = {
            "obj_func": tt.objective_function,
            "bounds": bounds,
            "minmax": "max",
            "verbose": False,
        }
        model = GA.BaseGA(epoch=5, pop_size=10, pc=0.9, pm=0.01)
        model.solve(problem, seed=42)

        assert model.g_best.target.fitness > 0
        schedule = tt.get_heuristic_schedule()
        assert sum(schedule) > 0
        assert len(schedule) == tt.n_services

    def test_update_supply(self, supply, revenue_behavior, supply_path):
        from mealpy import FloatVar, GA

        tt = MealpyTimetabling(
            requested_services=supply.services,
            revenue_behavior=revenue_behavior,
            safe_headway=10,
            max_stop_time=10,
        )
        bounds = [FloatVar(lb=lb, ub=ub) for lb, ub in tt.boundaries.real]
        problem = {
            "obj_func": tt.objective_function,
            "bounds": bounds,
            "minmax": "max",
            "verbose": False,
        }
        model = GA.BaseGA(epoch=5, pop_size=10, pc=0.9, pm=0.01)
        model.solve(problem, seed=42)

        best_position = model.g_best.solution
        schedule = tt.get_heuristic_schedule()
        solution = Solution(real=best_position, discrete=schedule)
        updated = tt.update_supply(str(supply_path), solution)
        assert len(updated) == int(sum(schedule))


class TestGSAE2E:
    def test_gsa_smoke(self, supply, revenue_behavior):
        tt = MealpyTimetabling(
            requested_services=supply.services,
            revenue_behavior=revenue_behavior,
            safe_headway=10,
            max_stop_time=10,
        )
        n_real = len(tt.boundaries.real)
        bounds = Boundaries(real=tt.boundaries.real, discrete=[])

        def gsa_obj(sol):
            sol_arr = np.array(sol.real, dtype=np.int32)
            tt.schedule_manager.update_from_solution(sol_arr)
            tt.revenue_calculator.updated_schedule = (
                tt.schedule_manager.updated_schedule
            )
            tt.revenue_calculator.recompute_all_revenues()
            schedule = tt.get_heuristic_schedule()
            revenue = tt.get_revenue(Solution(real=sol.real, discrete=schedule))
            accuracy = float(np.sum(schedule)) / len(schedule)
            return revenue, accuracy

        gsa = GSA(objective_function=gsa_obj, r_dim=n_real, d_dim=0, boundaries=bounds)
        gsa.optimize(population_size=10, iters=5, seed=42, verbose=False)

        assert gsa.best_fitness > float("-inf")
        schedule = tt.get_heuristic_schedule()
        assert sum(schedule) > 0
        assert len(schedule) == tt.n_services
