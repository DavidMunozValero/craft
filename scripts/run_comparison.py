#!/usr/bin/env python
"""Run the method x seed comparison grid on one priced instance.

Loads the notebook-0.0 instance (``timetable.yaml`` + ``revenue.yaml``), runs
the selected methods for every seed with a shared evaluation budget, measures
every solution with the common evaluator (:mod:`craft.ga.evaluation`) and
writes structured artifacts under ``--output-dir/<instance tag>/``:

* ``runs.csv`` — one row per (method, seed) with all common metrics;
* ``summary.csv`` — mean +- std per method;
* ``indicators_nsga2.csv`` — normalized hypervolume, IGD and spread of
  every NSGA-II front against the union of fronts (per seed);
* ``convergence_{method}_seed{seed}.csv``;
* ``pareto_front_{method}_seed{seed}.csv`` (NSGA-II front, measured);
* ``best_schedule_{method}_seed{seed}.csv`` (knee schedule for NSGA-II).

Examples:
    python scripts/run_comparison.py --methods all --seeds 7,21,42,63,84
    python scripts/run_comparison.py --methods ga_direct,nsga2 --epoch 100
    python scripts/run_comparison.py --fair-indices jain,gini,atkinson
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd

from craft.common import Solution
from craft.ga import (
    GA,
    NSGA2,
    SolutionEvaluator,
    load_problem,
    revenue_fairness_objectives,
)
from craft.ga.indicators import (
    ideal,
    igd,
    min_max_normalize,
    nadir,
    normalized_hypervolume,
    spread,
)
from craft.ga.nsga2 import knee_index
from craft.ga.objectives import FAIRNESS_ALPHA
from craft.gsa import GSA
from craft.mealpy import MealpyTimetabling

METRICS = [
    "revenue",
    "penalized",
    "planned",
    "conflicts",
    "fairness_jain",
    "fairness_gini",
    "fairness_atkinson",
    "inequity",
    "time",
]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the bi-objective comparison grid on one priced instance."
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
        "--methods",
        type=str,
        default="all",
        help=(
            "Comma-separated subset of "
            "greedy,ga_mealpy,ga_mealpy_fair,gsa,ga_direct,nsga2 "
            "(default: all)"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default="7,21,42,63,84",
        help="Comma-separated random seeds (default: 7,21,42,63,84)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=40, help="Population size (default: 40)"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="Generations/iterations (default: 100)"
    )
    parser.add_argument(
        "--pc", type=float, default=0.9, help="Crossover probability (default: 0.9)"
    )
    parser.add_argument(
        "--pm",
        type=float,
        default=0.05,
        help="Direct-encoding mutation probability (default: 0.05)",
    )
    parser.add_argument(
        "--pm-mealpy",
        type=float,
        default=0.01,
        help="mealpy GA mutation probability (default: 0.01)",
    )
    parser.add_argument(
        "--lambda",
        dest="conflict_penalty",
        type=float,
        default=2.0,
        help="Conflict penalty weight lambda (default: 2.0)",
    )
    parser.add_argument(
        "--max-stop-time",
        type=int,
        default=10,
        help="Maximum dwell-time extension in minutes (default: 10)",
    )
    parser.add_argument(
        "--fair-indices",
        type=str,
        default="jain",
        help=(
            "Comma-separated fairness indices for the rev*fair scalar sweep "
            "of the second paper (default: jain). Each index adds a "
            "ga_mealpy_fair_<index> method with its calibrated alpha."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/benchmark"),
        help="Directory for the benchmark artifacts (default: data/results/benchmark)",
    )
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    return parser


def run_greedy(problem):
    """Greedy heuristic of the paper at the requested timetables."""
    start = time.time()
    kept, _ = problem.greedy_baseline()
    mask = np.array([sid in set(kept) for sid in problem.service_ids])
    real = problem.reference_departures.astype(float)
    elapsed = time.time() - start
    return real, mask, None, elapsed


def run_ga_mealpy(seed, tt, epoch, pop_size, pc, pm):
    """mealpy GA over the real departure variables (heuristic CaSP mask)."""
    from mealpy import FloatVar
    from mealpy import GA as MealpyGA

    bounds = [FloatVar(lb=lb, ub=ub) for lb, ub in tt.boundaries.real]
    problem = {
        "obj_func": tt.objective_function,
        "bounds": bounds,
        "minmax": "max",
        "verbose": False,
    }
    model = MealpyGA.BaseGA(epoch=epoch, pop_size=pop_size, pc=pc, pm=pm)
    start = time.time()
    model.solve(problem, seed=seed)
    elapsed = time.time() - start
    real = np.asarray(model.g_best.solution, dtype=np.int32)
    mask = tt.get_heuristic_schedule()
    convergence = np.array([d.target.fitness for d in model.history.list_global_best])
    return real, mask, convergence, elapsed


def run_gsa(seed, tt, epoch, pop_size):
    """GSA with repair (deltas -> cumulative shifts) and seeded population."""
    from craft.common import Boundaries

    boundaries = Boundaries(real=tt.boundaries.real, discrete=[])

    def gsa_objective(sol):
        sol_arr = np.array(sol.real, dtype=np.int32)
        tt.schedule_manager.update_from_solution(sol_arr)
        tt.revenue_calculator.updated_schedule = dict(
            tt.schedule_manager.updated_schedule
        )
        tt.revenue_calculator.recompute_all_revenues()
        schedule = tt.get_heuristic_schedule()
        revenue = tt.get_revenue(Solution(real=sol.real, discrete=schedule))
        accuracy = float(np.sum(schedule)) / len(schedule)
        return revenue, accuracy

    def repair(solution):
        return Solution(
            real=tt.repair_departures(solution.real),
            discrete=np.array(solution.discrete, dtype=int),
        )

    np.random.seed(seed)
    requested = np.array(tt.reference_solution, dtype=float)
    lower = np.array([pair[0] for pair in boundaries.real], dtype=float)
    upper = np.array([pair[1] for pair in boundaries.real], dtype=float)
    draws = np.random.uniform(low=lower, high=upper, size=(pop_size - 1, len(lower)))
    initial = [
        Solution(real=tt.repair_departures(requested), discrete=np.zeros(0, dtype=int))
    ]
    initial.extend(
        Solution(real=tt.repair_departures(draw), discrete=np.zeros(0, dtype=int))
        for draw in draws
    )

    gsa = GSA(
        objective_function=gsa_objective,
        r_dim=len(boundaries.real),
        d_dim=0,
        boundaries=boundaries,
        is_feasible=lambda solution: False,
        custom_repair=repair,
    )
    start = time.time()
    gsa.optimize(
        population_size=pop_size,
        iters=epoch,
        seed=seed,
        repair_solution=True,
        initial_population=initial,
        verbose=False,
    )
    elapsed = time.time() - start
    real = tt.repair_departures(gsa.best_solution.real)
    mask = tt.get_heuristic_schedule()
    return real, mask, np.asarray(gsa.convergence, dtype=float), elapsed


def run_ga_direct(seed, problem, evaluator, epoch, pop_size, pc, pm):
    """Direct-encoding GA (mask + deltas, penalized revenue)."""
    solver = GA(problem, pop_size=pop_size, pc=pc, pm=pm)
    start = time.time()
    solver.optimize(epoch=epoch, seed=seed, verbose=False)
    elapsed = time.time() - start
    solution = problem.decode(solver.best_genome)
    return solution.real, solution.discrete, np.asarray(solver.convergence), elapsed


def run_nsga2(seed, problem, evaluator, epoch, pop_size, pc, pm):
    """Bi-objective NSGA-II on the revenue-Jain axis; returns knee and front."""
    objectives = revenue_fairness_objectives(problem, index="jain")
    solver = NSGA2(problem, pop_size=pop_size, pc=pc, pm=pm, objectives=objectives)
    start = time.time()
    solver.optimize(epoch=epoch, seed=seed, verbose=False)
    elapsed = time.time() - start

    front_rows = []
    for genome in solver.pareto_genomes:
        solution = problem.decode(genome)
        point = evaluator.evaluate(solution.real, solution.discrete)
        front_rows.append(point)
    front = pd.DataFrame(front_rows)
    knee = knee_index(front[["penalized", "fairness_jain"]].to_numpy())
    knee_solution = problem.decode(solver.pareto_genomes[knee])
    convergence = np.asarray(solver.convergence, dtype=float)
    return knee_solution.real, knee_solution.discrete, convergence, elapsed, front, knee


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    fair_indices = [index.strip() for index in args.fair_indices.split(",")]
    for index in fair_indices:
        if index not in FAIRNESS_ALPHA:
            parser.error(f"Unknown fairness index: {index!r}")

    base_methods = ["greedy", "ga_mealpy", "gsa", "ga_direct", "nsga2"]
    if args.methods == "all":
        methods = base_methods + [
            f"ga_mealpy_fair_{index}" for index in fair_indices
        ]
    else:
        methods = [name.strip() for name in args.methods.split(",")]
    known = set(base_methods) | {
        f"ga_mealpy_fair_{index}" for index in fair_indices
    }
    unknown = set(methods) - known
    if unknown:
        parser.error(f"Unknown methods: {sorted(unknown)}")
    seeds = [int(seed) for seed in args.seeds.split(",")]

    tag = f"{args.timetable.parent.name}_{args.timetable.stem}"
    output_dir = args.output_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    problem = load_problem(
        args.timetable,
        args.revenue,
        max_stop_time=args.max_stop_time,
        conflict_penalty=args.conflict_penalty,
    )
    evaluator = SolutionEvaluator(problem, conflict_penalty=args.conflict_penalty)
    import yaml

    revenue_behavior = yaml.safe_load(args.revenue.read_text())["services"]
    tt_mono = MealpyTimetabling(
        requested_services=problem.services,
        revenue_behavior=revenue_behavior,
        safe_headway=problem.safe_headway,
        max_stop_time=problem.max_stop_time,
    )
    tt_fairs = {
        index: MealpyTimetabling(
            requested_services=problem.services,
            revenue_behavior=revenue_behavior,
            safe_headway=problem.safe_headway,
            max_stop_time=problem.max_stop_time,
            fair_index=index.capitalize(),
            alpha=FAIRNESS_ALPHA[index],
        )
        for index in fair_indices
    }

    rows = []
    fronts = {}
    greedy_done = False
    for seed in seeds:
        for method in methods:
            if method == "greedy":
                if greedy_done:
                    continue
                greedy_done = True
                real, mask, convergence, elapsed = run_greedy(problem)
                front = None
                knee = None
            elif method == "ga_mealpy":
                real, mask, convergence, elapsed = run_ga_mealpy(
                    seed, tt_mono, args.epoch, args.pop_size, args.pc, args.pm_mealpy
                )
                front = None
                knee = None
            elif method.startswith("ga_mealpy_fair_"):
                real, mask, convergence, elapsed = run_ga_mealpy(
                    seed,
                    tt_fairs[method.removeprefix("ga_mealpy_fair_")],
                    args.epoch,
                    args.pop_size,
                    args.pc,
                    args.pm_mealpy,
                )
                front = None
                knee = None
            elif method == "gsa":
                real, mask, convergence, elapsed = run_gsa(
                    seed, tt_mono, args.epoch, args.pop_size
                )
                front = None
                knee = None
            elif method == "ga_direct":
                real, mask, convergence, elapsed = run_ga_direct(
                    seed,
                    problem,
                    evaluator,
                    args.epoch,
                    args.pop_size,
                    args.pc,
                    args.pm,
                )
                front = None
                knee = None
            else:
                real, mask, convergence, elapsed, front, knee = run_nsga2(
                    seed,
                    problem,
                    evaluator,
                    args.epoch,
                    args.pop_size,
                    args.pc,
                    args.pm,
                )
                fronts[seed] = front

            row = {"method": method, "seed": seed, "time": elapsed}
            row.update(evaluator.evaluate(real, mask))
            rows.append(row)

            if convergence is not None:
                pd.DataFrame(
                    {
                        "Iteration": np.arange(len(convergence)),
                        "Fitness": convergence,
                    }
                ).to_csv(
                    output_dir / f"convergence_{method}_seed{seed}.csv", index=False
                )
            if front is not None:
                front.to_csv(
                    output_dir / f"pareto_front_{method}_seed{seed}.csv", index=False
                )
            schedule_df = problem.schedule_frame(
                Solution(
                    real=np.asarray(real, dtype=float),
                    discrete=np.asarray(mask, dtype=bool),
                )
            )
            schedule_df.to_csv(
                output_dir / f"best_schedule_{method}_seed{seed}.csv", index=False
            )
            if not args.quiet:
                print(
                    f"seed {seed} | {method}: revenue={row['revenue']:.2f}, "
                    f"penalized={row['penalized']:.2f}, planned={row['planned']}, "
                    f"conflicts={row['conflicts']}, Jain={row['fairness_jain']:.4f}, "
                    f"t={elapsed:.1f}s"
                )

    runs = pd.DataFrame(rows)
    runs_path = output_dir / "runs.csv"
    runs.to_csv(runs_path, index=False)

    summary = {}
    for method, group in runs.groupby("method", sort=False):
        summary[method] = {}
        for metric in METRICS:
            mean = group[metric].mean()
            std = group[metric].std()
            if mean != mean:
                summary[method][metric] = "—"
            elif std == std:
                summary[method][metric] = f"{mean:.2f} ± {std:.2f}"
            else:
                summary[method][metric] = f"{mean:.2f}"
    summary_df = pd.DataFrame(summary).T
    summary_path = output_dir / "summary.csv"
    summary_df.to_csv(summary_path, sep=";")

    if fronts:
        objective_columns = ["penalized", "fairness_jain"]
        union = np.vstack(
            [front[objective_columns].to_numpy() for front in fronts.values()]
        )
        low, high = nadir(union), ideal(union)
        normalized_union = min_max_normalize(union, low, high)
        indicator_rows = []
        for seed, front in fronts.items():
            points = min_max_normalize(
                front[objective_columns].to_numpy(), low, high
            )
            indicator_rows.append(
                {
                    "seed": seed,
                    "n_points": len(points),
                    "hypervolume": normalized_hypervolume(points, normalized_union),
                    "igd": igd(points, normalized_union),
                    "spread": spread(points, normalized_union),
                }
            )
        indicators_path = output_dir / "indicators_nsga2.csv"
        pd.DataFrame(indicator_rows).to_csv(indicators_path, index=False)
        print(f"Indicadores MO -> {indicators_path}")

    print(f"\nRuns    -> {runs_path}")
    print(f"Summary -> {summary_path}")
    print(f"Artefactos por run en {output_dir}/")


if __name__ == "__main__":
    main()
