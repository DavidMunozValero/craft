"""Tests for the MealpyTimetabling problem head (multiplicative objective)."""

import numpy as np
import pytest

from craft.common import Solution
from craft.fairness import FairnessMetrics
from craft.mealpy import MealpyTimetabling


@pytest.fixture()
def timetabling(supply, revenue_behavior) -> MealpyTimetabling:
    """Plain revenue-maximization problem over the session fixture supply."""
    return MealpyTimetabling(
        requested_services=supply.services,
        revenue_behavior=revenue_behavior,
        safe_headway=10,
        max_stop_time=10,
    )


@pytest.fixture()
def fair_timetabling(supply, revenue_behavior) -> MealpyTimetabling:
    """Multiplicative revenue x Jain(25) problem over the same supply."""
    return MealpyTimetabling(
        requested_services=supply.services,
        revenue_behavior=revenue_behavior,
        safe_headway=10,
        max_stop_time=10,
        fair_index="Jain",
        alpha=25.0,
    )


def test_objective_function_is_revenue_by_default(timetabling):
    """Without a fair index the objective stays the raw revenue."""
    solution = list(timetabling.reference_solution)
    schedule = timetabling.get_heuristic_schedule()
    expected = timetabling.get_revenue(Solution(real=list(solution), discrete=schedule))
    assert timetabling.objective_function(list(solution)) == pytest.approx(expected)


def test_multiplicative_objective_matches_revenue_times_jain(
    timetabling, fair_timetabling
):
    """The fairness objective multiplies the revenue by the capacity-aware
    Jain index of the heuristic schedule."""
    solution = list(timetabling.reference_solution)
    revenue = timetabling.objective_function(list(solution))
    fair_value = fair_timetabling.objective_function(list(solution))

    schedule = timetabling.get_heuristic_schedule()
    expected_fairness, _ = FairnessMetrics.jains_fairness_index(
        schedule, timetabling.capacities, timetabling.revenue, alpha=25.0
    )
    assert fair_value == pytest.approx(revenue * expected_fairness)


def test_alpha_changes_the_multiplicative_objective(supply, revenue_behavior):
    """A higher sensitivity exponent lowers the objective on unequal masks."""
    plain = MealpyTimetabling(
        requested_services=supply.services, revenue_behavior=revenue_behavior
    )
    mild = MealpyTimetabling(
        requested_services=supply.services,
        revenue_behavior=revenue_behavior,
        fair_index="Jain",
        alpha=1.0,
    )
    harsh = MealpyTimetabling(
        requested_services=supply.services,
        revenue_behavior=revenue_behavior,
        fair_index="Jain",
        alpha=25.0,
    )
    solution = list(plain.reference_solution)
    mild_value = mild.objective_function(list(solution))
    harsh_value = harsh.objective_function(list(solution))
    revenue = plain.objective_function(list(solution))
    assert 0.0 < mild_value <= revenue
    assert harsh_value <= mild_value


def test_gini_and_atkinson_are_supported(supply, revenue_behavior):
    """Every Fair-oriented index drives the multiplicative objective."""
    solution = None
    for index in ("Gini", "Atkinson"):
        tt = MealpyTimetabling(
            requested_services=supply.services,
            revenue_behavior=revenue_behavior,
            fair_index=index,
            alpha=10.0 if index == "Gini" else 25.0,
        )
        solution = list(tt.reference_solution)
        value = tt.objective_function(solution)
        assert value >= 0.0


def test_gini_and_atkinson_score_zero_when_a_ru_is_left_out(supply, revenue_behavior):
    """Gini and Atkinson demand full RU coverage; partial allocations are 0."""
    first_ru = next(iter(revenue_behavior.values()))["ru"]
    partial = np.array(
        [behavior["ru"] == first_ru for behavior in revenue_behavior.values()]
    )
    assert not partial.all()
    for fair_index in ("Gini", "Atkinson"):
        tt = MealpyTimetabling(
            requested_services=supply.services,
            revenue_behavior=revenue_behavior,
            fair_index=fair_index,
        )
        assert tt.get_fairness(partial) == 0.0


def test_unknown_fair_index_is_rejected(supply, revenue_behavior):
    tt = MealpyTimetabling(
        requested_services=supply.services,
        revenue_behavior=revenue_behavior,
        fair_index="Theil",
    )
    with pytest.raises(ValueError):
        tt.get_fairness(tt.get_heuristic_schedule())


def test_empty_schedule_fairness_is_one(timetabling):
    """The null plan scores perfect fairness (never wins with zero revenue)."""
    empty = np.zeros(timetabling.n_services, dtype=bool)
    assert timetabling.get_fairness(empty) == 1.0


class TestRepairDepartures:
    def test_reference_is_a_fixed_point(self, timetabling):
        """The requested timetable repairs to itself."""
        requested = np.array(timetabling.reference_solution, dtype=float)
        repaired = timetabling.repair_departures(requested)
        assert np.allclose(repaired, requested)

    def test_matches_the_direct_encoding_decode(
        self, timetabling, supply, revenue_behavior
    ):
        """The repair reproduces the direct-encoding decode of the same
        deltas, so both backends explore the identical feasible set."""
        from craft.ga.timetabling import GATimetabling

        problem = GATimetabling(supply, revenue_behavior)
        rng = np.random.default_rng(7)
        genome = np.zeros(problem.genome_size, dtype=np.int64)
        for block in problem.slices:
            width = block.stop - block.start
            genome[block] = rng.integers(
                problem.lower_bounds[block],
                problem.upper_bounds[block] + 1,
                size=width,
            )
            genome[block.start] = 1

        solution = problem.decode(genome)
        repaired = timetabling.repair_departures(solution.real)
        assert np.allclose(repaired, solution.real)

    def test_clamps_origin_beyond_margin(self, timetabling):
        """An origin pushed far beyond the IM margin comes back inside it."""
        requested = np.array(timetabling.reference_solution, dtype=float)
        shifted = requested.copy()
        shifted[0] += 500.0
        repaired = timetabling.repair_departures(shifted)
        assert repaired[0] == requested[0] + timetabling.im_mod_margin

    def test_intermediate_delay_propagates_without_compressing_times(self, timetabling):
        """A delayed origin pushes the whole service downstream: the shifts
        never decrease, so travel and stop times cannot shrink."""
        tt = timetabling
        first_service = next(iter(tt.reference_schedules))
        schedule = tt.reference_schedules[first_service]
        n_times = len(schedule) - 1

        offset = 0
        for sid in tt.reference_schedules:
            if sid == first_service:
                break
            offset += len(tt.reference_schedules[sid]) - 1

        real = np.array(tt.reference_solution, dtype=float)
        real[offset] += 10.0
        repaired = tt.repair_departures(real)

        original_departures = np.array(
            [schedule[stops][1] for stops in schedule], dtype=float
        )[:n_times]
        shifts = repaired[offset : offset + n_times] - original_departures
        assert np.all(np.diff(shifts) >= 0)
        assert shifts[0] == 10.0

    def test_output_is_operationally_feasible(self, timetabling):
        """Every repaired service passes the operational feasibility filter."""
        requested = np.array(timetabling.reference_solution, dtype=float)
        rng = np.random.default_rng(3)
        noisy = requested + rng.uniform(-40.0, 40.0, size=requested.shape)
        repaired = timetabling.repair_departures(noisy)

        tt = timetabling
        tt.schedule_manager.update_from_solution(repaired)
        assert all(
            tt.schedule_manager.is_service_feasible(sid)
            for sid in tt.reference_schedules
        )


def test_heuristic_schedule_breaks_ties_by_service_id(timetabling):
    """Equal revenues make the CaSP heuristic keep the lowest service id of
    a conflicting pair, so the mask is reproducible across processes."""
    from craft.common import ConflictMatrix

    tt = timetabling
    tt.revenue_calculator.get_service_revenue = lambda sid: 100.0
    ids = list(tt.reference_schedules)
    first, second = ids[0], ids[1]
    tt.conflict_matrix = ConflictMatrix(services=tt.updated_services)
    tt.conflict_matrix.set(first, second, True)

    mask = tt.get_heuristic_schedule()
    assert mask[tt.indexer[first]]
    assert not mask[tt.indexer[second]]
