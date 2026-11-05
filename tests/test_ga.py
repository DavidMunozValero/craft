"""Tests for the direct genetic encoding (``craft.ga.timetabling``)."""

from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import yaml
from robin.supply.entities import Supply
from robin.supply.generator.entities import ServiceScheduler

from craft.ga.timetabling import GATimetabling, cumulative_shifts, load_problem
from craft.scheduling import get_schedule_from_supply

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SYNTHETIC_DIR = PROJECT_ROOT / "data" / "synthetic"


@pytest.fixture()
def problem(supply, revenue_behavior) -> GATimetabling:
    """Timetabling problem over the session fixture supply."""
    return GATimetabling(supply, revenue_behavior)


def test_cumulative_shifts_example():
    """Deltas (5, 0, 1) accumulate downstream into shifts (5, 5, 6)."""
    shifts = cumulative_shifts(
        deltas=np.array([5, 0, 1]),
        dwells=np.array([2, 2]),
        im_mod_margin=60,
        max_stop_time=10,
    )
    assert shifts.tolist() == [5, 5, 6]


def test_cumulative_shifts_clamps_genes():
    """Genes outside their bounds are clamped: the origin delta to the IM
    margin and downstream shifts to the dwell cap."""
    shifts = cumulative_shifts(
        deltas=np.array([100, 50, 5, 5]),
        dwells=np.array([2, 2, 2]),
        im_mod_margin=60,
        max_stop_time=10,
    )
    assert shifts.tolist() == [60, 60, 60, 60]

    shifts = cumulative_shifts(
        deltas=np.array([8, 5, 5, 5]),
        dwells=np.array([2, 2, 2]),
        im_mod_margin=60,
        max_stop_time=10,
    )
    assert shifts.tolist() == [8, 10, 10, 10]


def test_cumulative_shifts_allow_advancing():
    """A negative origin shift propagates downstream without ever shrinking
    travel or stop times."""
    shifts = cumulative_shifts(
        deltas=np.array([-30, 5, 5]),
        dwells=np.array([2, 2]),
        im_mod_margin=60,
        max_stop_time=10,
    )
    assert shifts.tolist() == [-30, -25, -20]


def test_decode_applies_cumulative_shifts(supply, revenue_behavior):
    """Decoding translates a planned block into absolute departure times."""
    tt = GATimetabling(supply, revenue_behavior)
    sid = tt.service_ids[0]
    n_times = len(tt.reference_schedules[sid]) - 1

    genome = np.zeros(tt.genome_size, dtype=np.int64)
    for block in tt.slices:
        genome[block.start] = 1
    deltas = np.zeros(n_times, dtype=np.int64)
    deltas[0] = 5
    if n_times > 2:
        deltas[2] = 1
    genome[tt.time_slices[0]] = deltas

    solution = tt.decode(genome)
    expected = cumulative_shifts(
        deltas, tt.dwells[0], tt.im_mod_margin, tt.max_stop_time
    )
    reference = np.array(
        [
            tt.reference_schedules[sid][stop][1]
            for stop in list(tt.reference_schedules[sid])[:-1]
        ]
    )
    assert solution.discrete.all()
    assert solution.real[tt.real_slices[0]].tolist() == (reference + expected).tolist()


def test_decode_keeps_unplanned_services(supply, revenue_behavior):
    """Services with b = 0 keep their requested timetable."""
    tt = GATimetabling(supply, revenue_behavior)
    genome = np.zeros(tt.genome_size, dtype=np.int64)
    genome[tt.slices[0].start] = 1

    solution = tt.decode(genome)
    assert solution.discrete[0]
    assert solution.discrete[1:].sum() == 0
    assert (
        solution.real[tt.real_slices[1]].tolist()
        == tt.reference_departures[tt.real_slices[1]].tolist()
    )


def test_decode_is_feasible_by_construction(supply, revenue_behavior):
    """Random genomes decode into schedules that honor margins, dwell caps and
    operating times for every planned service."""
    tt = GATimetabling(supply, revenue_behavior)
    rng = np.random.default_rng(42)
    for _ in range(50):
        genome = rng.integers(-120, 120, size=tt.genome_size)
        solution = tt.decode(genome)

        tt.schedule_manager.update_from_solution(solution.real)
        for i, sid in enumerate(tt.service_ids):
            reference = tt.reference_schedules[sid]
            times = solution.real[tt.real_slices[i]]
            stops = list(reference)[:-1]
            for j, (stop, value) in enumerate(zip(stops, times)):
                cap = tt.upper_bounds[tt.slices[i].start + 1 + j]
                assert reference[stop][1] - tt.im_mod_margin <= value
                assert value <= reference[stop][1] + cap
            if solution.discrete[i]:
                assert tt.schedule_manager.is_service_feasible(sid)
            else:
                assert times.tolist() == [reference[stop][1] for stop in stops]


def test_conflict_checker_matches_robin(supply, revenue_behavior):
    """The corridor checker reproduces robin's conflicts on the reference
    schedules."""
    tt = GATimetabling(supply, revenue_behavior)
    schedules = dict(tt.schedule_manager.updated_schedule)
    matrix = tt.conflict_matrix(schedules)
    mine = {
        tuple(sorted((tt.service_ids[a], tt.service_ids[b])))
        for a, b in zip(*np.nonzero(np.triu(matrix, 1)))
    }
    assert mine == robin_conflicts(supply)


def robin_conflicts(supply: Supply, headway: int = 10) -> set:
    """Conflicting service-id pairs according to robin, without self-pairs."""
    scheduler = ServiceScheduler(services=supply.services)
    return {
        tuple(sorted((service.id, other)))
        for service in supply.services
        for other in scheduler.find_conflicts(new_service=service, safety_gap=headway)
        if other != service.id
    }


def test_fitness_penalizes_conflicts_and_respects_mask(supply, revenue_behavior):
    """Fitness is 0 without planned services and, for zero-delta genomes, it
    equals the canon of the planned services minus the conflict penalty. The
    session fixture is conflict-free, so the penalty term vanishes there; the
    synthetic-instance test exercises the penalty path."""
    tt = GATimetabling(supply, revenue_behavior)
    conflicts = robin_conflicts(supply)

    def analytic(mask: np.ndarray) -> float:
        canon = {sid: params["canon"] for sid, params in tt.revenue_behavior.items()}
        planned = {sid for i, sid in enumerate(tt.service_ids) if mask[i]}
        penalty = sum(
            min(canon[a], canon[b])
            for a, b in conflicts
            if a in planned and b in planned
        )
        return sum(canon[sid] for sid in planned) - penalty

    assert tt.fitness(np.zeros(tt.genome_size, dtype=np.int64)) == 0.0

    full = np.ones(tt.n_services, dtype=bool)
    fitness_full = tt.fitness(planned_genome(tt, full))
    assert fitness_full == pytest.approx(analytic(full))
    assert fitness_full <= sum(
        params["canon"] for params in tt.revenue_behavior.values()
    )

    conflicted = {sid for pair in conflicts for sid in pair}
    trimmed = np.array([sid not in conflicted for sid in tt.service_ids], dtype=bool)
    fitness_trimmed = tt.fitness(planned_genome(tt, trimmed))
    assert fitness_trimmed == pytest.approx(analytic(trimmed))


def planned_genome(tt: GATimetabling, mask: np.ndarray) -> np.ndarray:
    """Zero-delta genome with the given planned mask."""
    genome = np.zeros(tt.genome_size, dtype=np.int64)
    for i, block in enumerate(tt.slices):
        genome[block.start] = int(mask[i])
    return genome


@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "timetable.yaml").exists(),
    reason="synthetic data not generated (run notebook 0.0)",
)
def test_load_problem_builds_the_synthetic_instance():
    """The synthetic instance (notebook 0.0) builds a problem whose fitness
    matches the analytic value over robin's conflicts."""
    tt = load_problem(SYNTHETIC_DIR / "timetable.yaml", SYNTHETIC_DIR / "revenue.yaml")
    supply = Supply.from_yaml(path=str(SYNTHETIC_DIR / "timetable.yaml"))
    revenue = yaml.safe_load((SYNTHETIC_DIR / "revenue.yaml").read_text())["services"]
    assert tt.service_ids == [service.id for service in supply.services]
    assert tt.safe_headway == 10
    assert tt.im_mod_margin == 60
    assert tt.genome_size == tt.n_services + len(tt.reference_departures)

    genome = planned_genome(tt, np.ones(tt.n_services, dtype=bool))
    canon = {sid: params["canon"] for sid, params in revenue.items()}
    conflicts = robin_conflicts(supply)
    expected = sum(canon.values()) - tt.conflict_penalty * sum(
        min(canon[a], canon[b]) for a, b in conflicts
    )
    assert tt.fitness(genome) == pytest.approx(expected)

    schedules = get_schedule_from_supply(supply=supply)
    matrix = tt.conflict_matrix(schedules)
    mine = {
        tuple(sorted((tt.service_ids[a], tt.service_ids[b])))
        for a, b in zip(*np.nonzero(np.triu(matrix, 1)))
    }
    assert mine == conflicts


def test_requested_schedules_stay_pristine(problem):
    """Evaluations mutate the working schedules but never the requested ones."""
    requested = deepcopy(problem.requested_schedules)
    rng = np.random.default_rng(3)
    for _ in range(5):
        problem.fitness(rng.integers(-10, 10, size=problem.genome_size))
    assert problem.requested_schedules == requested


def test_reporting_helpers(problem):
    """decoded_schedule, describe_genome, schedule_frame and greedy_baseline
    report consistent views of a zero-delta planned genome."""
    genome = planned_genome(problem, np.ones(problem.n_services, dtype=bool))

    schedule = problem.decoded_schedule(genome)
    assert set(schedule) == set(problem.service_ids)
    for sid, stops in schedule.items():
        reference = problem.requested_schedules[sid]
        for station, times in stops.items():
            assert times == list(reference[station])

    described = problem.describe_genome(genome)
    assert len(described) == problem.n_services
    assert (described["planned"] == 1).all()
    assert (
        described["deltas"]
        .apply(lambda deltas: bool(np.all(np.asarray(deltas) == 0)))
        .all()
    )

    frame = problem.schedule_frame(problem.decode(genome))
    assert (frame["origin_delta"] == 0).all()
    np.testing.assert_allclose(frame["revenue"].to_numpy(), frame["canon"].to_numpy())

    kept, revenue = problem.greedy_baseline()
    assert 0 < len(kept) <= problem.n_services
    canon = {sid: params["canon"] for sid, params in problem.revenue_behavior.items()}
    assert revenue == pytest.approx(sum(canon[sid] for sid in kept))
    matrix = problem.conflict_matrix(problem.requested_schedules)
    rows = [problem.row_index[sid] for sid in kept]
    assert not matrix[np.ix_(rows, rows)].any()
