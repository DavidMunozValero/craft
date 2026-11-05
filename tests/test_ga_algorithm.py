"""Tests for the genetic algorithm (``craft.ga.algorithm``)."""

from pathlib import Path

import numpy as np
import pytest

from craft.common import Solution
from craft.ga import GA, GATimetabling
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


def test_pop_size_is_validated(problem):
    """Populations below the tournament threshold are rejected."""
    with pytest.raises(ValueError):
        GA(problem, pop_size=5)


def test_problem_satisfies_the_solver_contract(problem):
    """Any solver can consume the problem through the Problem protocol."""
    from craft.ga.base import Problem

    assert isinstance(problem, Problem)


def test_initial_population_is_planned_and_in_bounds(problem):
    """Individuals start with every service planned and genes in bounds."""
    ga = GA(problem, pop_size=12)
    population = ga._initial_population(np.random.default_rng(0))
    assert population.shape == (12, problem.genome_size)
    assert (population[:, ga._starts] == 1).all()
    assert (population >= problem.lower_bounds).all()
    assert (population <= problem.upper_bounds).all()


def test_crossover_exchanges_whole_blocks(problem):
    """Children take whole service blocks from one parent or the other."""
    ga = GA(problem, pop_size=12, pc=1.0)
    p1 = np.zeros(problem.genome_size, dtype=np.int64)
    p2 = np.ones(problem.genome_size, dtype=np.int64)
    c1, c2 = ga._crossover(p1, p2, np.random.default_rng(42))

    for block in problem.slices:
        for child in (c1, c2):
            assert child[block].tolist() in (p1[block].tolist(), p2[block].tolist())
        assert c1[block].tolist() != c2[block].tolist()
    assert (c1 != p1).any()
    assert (c2 != p2).any()


def test_mutation_keeps_bounds_and_bits(problem):
    """Aggressive mutation never leaves the bounds nor the binary domain."""
    ga = GA(problem, pop_size=12, pm=1.0, pr=0.5, sigma=10)
    rng = np.random.default_rng(1)
    genome = problem.upper_bounds.copy()
    for _ in range(100):
        genome = ga._mutate(genome, rng)
        assert (genome >= problem.lower_bounds).all()
        assert (genome <= problem.upper_bounds).all()
        assert set(genome[ga._starts].tolist()) <= {0, 1}


def test_optimize_is_deterministic_with_seed(problem):
    """Two runs with the same seed produce identical search trajectories
    (modulo wall-clock timings)."""
    columns = ["Iteration", "Fitness", "Planned", "Conflicts"]
    histories = [
        GA(problem, pop_size=12).optimize(epoch=5, seed=42, verbose=False)
        for _ in range(2)
    ]
    assert histories[0][columns].equals(histories[1][columns])


def test_optimize_history_and_attributes(problem):
    """The run exposes convergence, history and the decoded best solution."""
    ga = GA(problem, pop_size=12)
    history = ga.optimize(epoch=8, seed=42, verbose=False)

    assert list(history.columns) == [
        "Iteration",
        "Fitness",
        "Planned",
        "Conflicts",
        "ExecutionTime",
    ]
    assert len(history) == 9
    assert len(ga.convergence) == 9
    assert (np.diff(ga.convergence) >= 0).all()
    assert ga.best_genomes.shape == (9, problem.genome_size)
    assert (ga.best_genomes[-1] == ga.best_genome).all()
    assert isinstance(ga.best_solution, Solution)
    assert ga.best_solution.discrete.sum() == history["Planned"].iloc[-1]
    assert ga.execution_time > 0.0


@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "timetable.yaml").exists(),
    reason="synthetic data not generated (run notebook 0.0)",
)
def test_optimize_recovers_revenue_on_synthetic():
    """On the conflicting synthetic instance the GA plans a profitable,
    conflict-free subset, beating the plan-everything baseline."""
    import yaml
    from robin.supply.entities import Supply

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
    ga = GA(problem, pop_size=40)
    history = ga.optimize(epoch=100, seed=42, verbose=False)

    assert ga.best_fitness > 0.0
    assert ga.best_fitness > baseline
    assert history["Conflicts"].iloc[0] > 0
    assert history["Conflicts"].iloc[-1] == 0
    assert ga.best_fitness > greedy_baseline()


def test_run_ga_script(tmp_path):
    """The CLI script runs end to end and writes its artifacts."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_ga.py"),
            "--seed",
            "7",
            "--pop-size",
            "12",
            "--epoch",
            "3",
            "--output-dir",
            str(tmp_path),
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr
    assert "GA results" in result.stdout
    for name in (
        "convergence_ga_seed7.csv",
        "best_schedule_ga_seed7.csv",
        "genome_evolution_ga_seed7.png",
        "timetable_evolution_ga_seed7.png",
    ):
        assert (tmp_path / name).exists()
