"""Tests for the epsilon-constraint reference model (``scripts/run_epsilon.py``)."""

import importlib.util
from pathlib import Path
from types import ModuleType

import pytest

pytest.importorskip("pyomo")
pytest.importorskip("pyscipopt")


from craft.ga import GATimetabling, SolutionEvaluator  # noqa: E402
from craft.ga.objectives import compute_capacities  # noqa: E402


def load_script(name: str) -> ModuleType:
    """Import a top-level script by file path."""
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def problem(supply, revenue_behavior) -> GATimetabling:
    """Timetabling problem over the session fixture supply."""
    return GATimetabling(supply, revenue_behavior)


@pytest.fixture(scope="module")
def evaluator(problem) -> SolutionEvaluator:
    return SolutionEvaluator(problem)


def solve(problem, spread: float):
    module = load_script("run_epsilon.py")
    model = module.build_model(problem, compute_capacities(problem.revenue_behavior), spread)
    return module.solve_model(model, time_limit=120.0)


def measure(problem, evaluator, mask):
    return evaluator.evaluate(problem.reference_departures.astype(float), mask)


def test_zero_spread_recovers_an_optimal_conflict_free_selection(problem, evaluator):
    """Without the balance restriction the exact model beats the greedy."""
    mask, condition, _bounds = solve(problem, spread=0.0)
    assert mask is not None, condition
    metrics = measure(problem, evaluator, mask)
    _, greedy_revenue = problem.greedy_baseline()
    assert metrics["revenue"] >= greedy_revenue - 1e-6


def test_solution_is_conflict_free_at_the_requested_schedules(problem, evaluator):
    """Conflict-freeness holds by construction over the requested timetables."""
    mask, condition, _bounds = solve(problem, spread=0.0)
    assert mask is not None, condition
    assert measure(problem, evaluator, mask)["conflicts"] == 0


def test_revenue_does_not_grow_with_the_balance_level(problem, evaluator):
    """Feasible sets nest in ``t``, so the max revenue is non-increasing."""
    free, condition, _bounds = solve(problem, spread=0.0)
    assert free is not None, condition
    constrained, condition, _bounds = solve(problem, spread=0.8)
    if constrained is None:
        pytest.skip(f"spread 0.8 infeasible on the fixture ({condition})")
    free_metrics = measure(problem, evaluator, free)
    constrained_metrics = measure(problem, evaluator, constrained)
    assert constrained_metrics["revenue"] <= free_metrics["revenue"] + 1e-6


def test_infeasible_spread_reports_its_condition(problem):
    """An impossible balance level is reported instead of crashing."""
    mask, condition, _bounds = solve(problem, spread=1.0)
    if mask is not None:
        pytest.skip(f"spread 1.0 turned out feasible ({condition})")
    assert "infeasible" in condition.lower()
