#!/usr/bin/env python
"""Exact epsilon-constraint reference points for the revenue-fairness plane.

Selection-level exact model over the requested timetables (the same role the
SCIP model plays in the first paper of the line): a binary planning mask that
is conflict-free by construction over the requested schedules, maximizing the
canon revenue under an allocation-balance restriction. The resulting points
enrich the reference front of the MO indicators (plan P4 / experiment E4).

Formulation — the balance parameter ``t`` bounds the relative spread of the
achieved-vs-capacity ratios: with ``a_r`` the importance mass planned for RU
``r`` and ``c_r`` its requested-service share, there must exist a common
``u`` with ``t * u <= a_r / c_r <= u`` for every RU. Every RU participates
and no RU's ratio exceeds ``1/t`` times another's, which is a sufficient
condition for the capacity-aware Jain floor ``Jain_alpha >= t**50`` at the
calibration alpha=25 (ratios equal within a factor t, raised to alpha,
squared in the index). The measured Jain of every point is reported by the
common evaluator, which is the same ruler the metaheuristics are measured
with.

Why not restrict the Jain index directly: the calibrated index raises the
ratios to ``alpha=25``, so the polynomial constraint evaluates to ~1e-7 for
typical allocations — below the tolerances of any branch-and-bound solver,
which then declares violated solutions optimal or prunes valid nodes (both
observed empirically). The linear balance restriction is an exact MILP per
level, its feasible sets are nested in ``t`` (so the revenue is exactly
monotone non-increasing), and infeasibility is certified by the solver.

Revenue is ``sum canon_i * B_i``: requested times imply zero deviation, so
the penalized and raw revenues coincide and conflicts are the static pairs
of the requested schedules. Time shifts are not part of this model; the
metaheuristic fronts may strictly dominate it by moving departures (that
difference is exactly the value of the direct encoding, made visible by the
common evaluator).

Examples:
    python scripts/run_epsilon.py
    python scripts/run_epsilon.py --timetable data/synthetic/instances/n25_r3/timetable.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import pyomo.environ as pyo

from craft.ga import SolutionEvaluator, load_problem
from craft.ga.objectives import compute_capacities

SOLVABLE_CONDITIONS = {
    pyo.TerminationCondition.optimal,
    pyo.TerminationCondition.maxTimeLimit,
    pyo.TerminationCondition.feasible,
}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Exact epsilon-constraint reference points (Pyomo + SCIP)."
    )
    parser.add_argument(
        "--timetable",
        type=Path,
        default=Path("data/synthetic/timetable.yaml"),
        help="Timetable YAML from notebook 0.0",
    )
    parser.add_argument(
        "--revenue",
        type=Path,
        default=Path("data/synthetic/revenue.yaml"),
        help="Revenue behavior YAML from notebook 0.0",
    )
    parser.add_argument(
        "--spreads",
        type=str,
        default="0,0.5,0.7,0.8,0.9,0.95,0.98,1.0",
        help=(
            "Comma-separated balance levels t: every RU ratio lies within "
            "[t*u, u] for a common u (default: 0,0.5,...,1.0)"
        ),
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="SCIP time limit per solve, seconds (default: 300)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/epsilon"),
        help="Directory for the reference points (default: data/results/epsilon)",
    )
    return parser


def build_model(problem, capacities, spread: float):
    """Selection-level balance-constrained model over the requested timetables.

    Args:
        problem: Direct-encoding problem head (service ids, requested
            schedules and revenue behavior).
        capacities: Capacity share (percent of requested services) per RU.
        spread: Balance level ``t``; 0 removes the balance restriction and 1
            forces exact proportionality between every RU ratio.

    Returns:
        The Pyomo model with binary planning variables ``B``, the canon
        revenue objective, the pairwise conflict-freeness constraints over
        the requested schedules and the linear balance restriction.
    """
    behavior = problem.revenue_behavior
    service_ids = list(problem.service_ids)
    canon = [float(behavior[sid]["canon"]) for sid in service_ids]
    importance = [float(behavior[sid].get("importance", 1.0)) for sid in service_ids]
    rus = sorted(capacities)
    shares = {ru: capacities[ru] / 100.0 for ru in rus}
    members = {ru: [] for ru in rus}
    for index, sid in enumerate(service_ids):
        members[behavior[sid]["ru"]].append(index)

    matrix = problem.conflict_matrix(problem.requested_schedules)
    pairs = [
        (i, j)
        for i in range(problem.n_services)
        for j in range(i + 1, problem.n_services)
        if matrix[i, j]
    ]

    model = pyo.ConcreteModel()
    model.B = pyo.Var(range(problem.n_services), domain=pyo.Binary)
    achieved = {}
    for ru in rus:
        achieved[ru] = sum(importance[i] * model.B[i] for i in members[ru])

    model.u = pyo.Var(
        domain=pyo.NonNegativeReals, bounds=(0.0, 1.0 / min(shares.values()))
    )
    model.revenue = pyo.Objective(
        expr=sum(canon[i] * model.B[i] for i in range(problem.n_services)),
        sense=pyo.maximize,
    )

    def no_conflict_rule(_model, i, j):
        return _model.B[i] + _model.B[j] <= 1

    model.no_conflict = pyo.Constraint(pairs, rule=no_conflict_rule)

    def ratio_upper_rule(_model, ru):
        return achieved[ru] <= _model.u * shares[ru]

    def ratio_lower_rule(_model, ru):
        return achieved[ru] >= spread * _model.u * shares[ru]

    model.ratio_upper = pyo.Constraint(rus, rule=ratio_upper_rule)
    model.ratio_lower = pyo.Constraint(rus, rule=ratio_lower_rule)
    return model


def solve_model(model, time_limit: float):
    """Solve with SCIP; return ``(mask, condition, bounds)``.

    ``mask`` is ``None`` when no solution was found (e.g. infeasible level).
    ``bounds`` reports the primal and dual bounds declared by the solver.
    """
    solver = pyo.SolverFactory("scip")
    solver.options["limits/time"] = time_limit
    results = solver.solve(model, load_solutions=False, tee=False)
    condition = results.solver.termination_condition
    if condition not in SOLVABLE_CONDITIONS:
        return None, str(condition), {}
    model.solutions.load_from(results)
    mask = np.array(
        [bool(round(pyo.value(model.B[i]))) for i in range(len(model.B))], dtype=bool
    )
    bounds = {}
    try:
        problem_result = results.problem[0]
        bounds = {
            "lower_bound": problem_result.lower_bound,
            "upper_bound": problem_result.upper_bound,
        }
    except (AttributeError, IndexError, KeyError):
        bounds = {}
    return mask, str(condition), bounds


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    spreads = [float(value) for value in args.spreads.split(",")]

    problem = load_problem(args.timetable, args.revenue)
    evaluator = SolutionEvaluator(problem)
    capacities = compute_capacities(problem.revenue_behavior)

    greedy_ids, greedy_revenue = problem.greedy_baseline()
    print(
        f"Instancia: {problem.n_services} servicios, {len(capacities)} RUs, "
        f"greedy = {len(greedy_ids)} trens / {greedy_revenue:.2f}\n"
    )

    tag = f"{args.timetable.parent.name}_{args.timetable.stem}"
    output_dir = args.output_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for spread in spreads:
        model = build_model(problem, capacities, spread)
        mask, condition, bounds = solve_model(model, args.time_limit)
        if mask is None:
            rows.append({"spread": spread, "status": condition})
            print(f"spread {spread:>5.2f}: {condition}")
            continue
        metrics = evaluator.evaluate(problem.reference_departures.astype(float), mask)
        row = {
            "spread": spread,
            "status": condition,
            "jain_floor": spread**50,
            "canon_objective": float(pyo.value(model.revenue)),
            **bounds,
            **metrics,
        }
        rows.append(row)
        print(
            f"spread {spread:>5.2f}: revenue {metrics['revenue']:.2f}, "
            f"Jain {metrics['fairness_jain']:.4f} (floor {spread**50:.2e}), "
            f"{metrics['planned']} trens, {metrics['conflicts']} conflictos "
            f"[{condition}]"
        )

    results = pd.DataFrame(rows)
    results_path = output_dir / "epsilon_reference.csv"
    results.to_csv(results_path, index=False)
    print(f"\nReferencia exacta -> {results_path}")


if __name__ == "__main__":
    main()
