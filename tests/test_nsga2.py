"""Tests for the NSGA-II backend (``craft.ga.nsga2``)."""

from pathlib import Path

import numpy as np
import pytest
from robin.supply.entities import Supply

from craft.common import Solution
from craft.ga import GATimetabling, NSGA2
from craft.ga.nsga2 import crowding_distance, dominates, fast_non_dominated_sort
from craft.scheduling import get_schedule_from_supply

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def planned_genome(problem: GATimetabling, mask: np.ndarray) -> np.ndarray:
    """Zero-delta genome with the given planned mask."""
    genome = np.zeros(problem.genome_size, dtype=np.int64)
    for i, block in enumerate(problem.slices):
        genome[block.start] = int(mask[i])
    return genome


@pytest.fixture()
def problem(supply, revenue_behavior) -> GATimetabling:
    """Timetabling problem over the session fixture supply."""
    return GATimetabling(supply, revenue_behavior)


def test_dominates_is_pareto_dominance():
    """Domination requires weakly better everywhere and better somewhere."""
    a = np.array([3.0, 4.0])
    assert dominates(a, np.array([1.0, 2.0]))
    assert not dominates(a, np.array([3.0, 4.0]))
    assert not dominates(a, np.array([5.0, 1.0]))


def test_fast_non_dominated_sort_known_case():
    """Fronts peel in domination order; mutual non-dominators share a front."""
    values = np.array([[3.0, 4.0], [1.0, 2.0], [3.0, 3.0], [2.0, 4.0]])
    fronts = fast_non_dominated_sort(values)
    assert [front.tolist() for front in fronts] == [[0], [2, 3], [1]]


def test_crowding_distance_boundaries_are_infinite():
    """Boundary individuals per objective get infinite distance."""
    values = np.array([[1.0], [2.0], [3.0], [4.0]])
    distance = crowding_distance(values)
    assert np.isinf(distance[[0, -1]]).all()
    assert distance[1] == pytest.approx(2.0 / 3.0)
    assert distance[2] == pytest.approx(2.0 / 3.0)


def test_crowding_distance_skips_degenerate_objectives():
    """Constant objectives add no distance and never divide by zero."""
    values = np.array([[2.0, 0.0], [2.0, 1.0], [2.0, 2.0]])
    distance = crowding_distance(values)
    assert np.isinf(distance[[0, -1]]).all()
    assert distance[1] == pytest.approx(1.0)


def test_environmental_selection_keeps_the_best():
    """With a single objective the mu+lambda selection keeps the fittest."""
    solver = NSGA2.__new__(NSGA2)
    solver.pop_size = 10
    merged = np.arange(24, dtype=np.int64).reshape(12, 2)
    values = np.arange(12, dtype=float).reshape(12, 1)
    _, survivors = solver._select(merged, values)
    assert survivors[:10, 0].tolist() == list(range(11, 1, -1))


def test_pop_size_is_validated(problem):
    """Populations below the tournament threshold are rejected."""
    with pytest.raises(ValueError):
        NSGA2(problem, pop_size=5)


def test_default_objectives_mirror_problem_fitness(problem):
    """Without custom objectives NSGA-II optimizes the GA penalized revenue."""
    solver = NSGA2(problem, pop_size=12)
    genome = planned_genome(problem, np.ones(problem.n_services, dtype=bool))
    values = solver._evaluate(genome.reshape(1, -1))
    assert values.shape == (1, 1)
    assert values[0, 0] == problem.fitness(genome)


def test_optimize_is_deterministic_with_seed(problem):
    """Two runs with the same seed produce identical search trajectories
    (modulo wall-clock timings)."""
    columns = ["Iteration", "Fitness", "Planned", "Conflicts"]
    histories = [
        NSGA2(problem, pop_size=12).optimize(epoch=5, seed=42, verbose=False)
        for _ in range(2)
    ]
    assert histories[0][columns].equals(histories[1][columns])


def test_optimize_history_and_attributes(problem):
    """The run exposes convergence, history, the Pareto front and the
    decoded best solution."""
    solver = NSGA2(problem, pop_size=12)
    history = solver.optimize(epoch=8, seed=42, verbose=False)

    assert list(history.columns) == [
        "Iteration",
        "Fitness",
        "Planned",
        "Conflicts",
        "ExecutionTime",
    ]
    assert len(history) == 9
    assert len(solver.convergence) == 9
    assert (np.diff(solver.convergence) >= 0).all()
    assert solver.best_genomes.shape == (9, problem.genome_size)
    assert (solver.best_genomes[-1] == solver.best_genome).all()
    assert isinstance(solver.best_solution, Solution)
    assert solver.best_solution.discrete.sum() == history["Planned"].iloc[-1]
    assert solver.pareto_values.ndim == 2
    assert solver.pareto_values.shape[1] == 1
    assert solver.pareto_values[0, 0] == solver.best_fitness
    assert solver.pareto_genomes.shape[1] == problem.genome_size
    assert solver.execution_time > 0.0


@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "timetable.yaml").exists(),
    reason="synthetic data not generated (run notebook 0.0)",
)
def test_optimize_recovers_revenue_on_synthetic():
    """On the conflicting synthetic instance NSGA-II plans a profitable,
    conflict-free subset, beating the plan-everything and greedy baselines."""
    import yaml

    supply = Supply.from_yaml(path=str(SYNTHETIC_DIR / "timetable.yaml"))
    revenue = yaml.safe_load((SYNTHETIC_DIR / "revenue.yaml").read_text())["services"]
    problem = GATimetabling(supply, revenue)

    canon = {sid: params["canon"] for sid, params in revenue.items()}

    def greedy_baseline() -> float:
        """Revenue of the published heuristic at the requested timetables:
        iteratively drop the cheapest service of every conflicting pair."""
        kept = list(problem.service_ids)
        reference = get_schedule_from_supply(supply=supply)
        matrix = problem.conflict_matrix(reference)
        while True:
            rows = [problem.row_index[sid] for sid in kept]
            sub = matrix[np.ix_(rows, rows)]
            if not sub.any():
                break
            a, b = next(zip(*np.nonzero(np.triu(sub, 1))))
            sa, sb = kept[a], kept[b]
            kept.remove(sa if canon[sa] < canon[sb] else sb)
        return sum(canon[sid] for sid in kept)

    baseline = problem.fitness(
        planned_genome(problem, np.ones(problem.n_services, dtype=bool))
    )
    solver = NSGA2(problem, pop_size=40)
    history = solver.optimize(epoch=100, seed=42, verbose=False)

    assert solver.best_fitness > 0.0
    assert solver.best_fitness > baseline
    assert history["Conflicts"].iloc[0] > 0
    assert history["Conflicts"].iloc[-1] == 0
    assert solver.best_fitness > greedy_baseline()
