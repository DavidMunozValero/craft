#!/usr/bin/env python
"""Cross-check the craft NSGA-II front against pymoo's reference implementation.

Loads a priced instance (notebook-0.0 format), runs the craft NSGA-II and the
pymoo NSGA-II over the same genome layout, objectives and evaluation budget,
measures both fronts with the common evaluator and compares them with the
quality indicators of :mod:`craft.ga.indicators` (set coverage both ways,
normalized hypervolume and IGD against the union front). The validation
criterion of the publication plan (P4) is ``c_metric`` without massive mutual
domination: neither front should cover the other while remaining uncovered.

pymoo minimizes, so the adapter negates the (maximization) objective vector;
its population lives on a continuous relaxation of the integer genome and is
rounded back to integers at evaluation time.

Examples:
    python scripts/run_pymoo.py --seeds 7,21,42
    python scripts/run_pymoo.py --timetable data/synthetic/instances/n25_r3/timetable.yaml
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from pymoo.algorithms.moo.nsga2 import NSGA2 as PymooNSGA2
from pymoo.core.problem import Problem as PymooProblem
from pymoo.optimize import minimize

from craft.ga import (
    NSGA2,
    SolutionEvaluator,
    load_problem,
    revenue_fairness_objectives,
)
from craft.ga.indicators import (
    c_metric,
    ideal,
    igd,
    min_max_normalize,
    nadir,
    normalized_hypervolume,
)

FRONT_COLUMNS = ["penalized", "fairness_jain"]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cross-check craft NSGA-II vs pymoo on one priced instance."
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
        "--seeds",
        type=str,
        default="7,21,42,63,84",
        help="Comma-separated random seeds (default: 7,21,42,63,84)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=40, help="Population size (default: 40)"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="Generations (default: 100)"
    )
    parser.add_argument(
        "--fair-index",
        choices=["jain", "gini", "atkinson"],
        default="jain",
        help="Fairness index of the second objective (default: jain)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/results/pymoo"),
        help="Directory for the cross-check artifacts (default: data/results/pymoo)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


class PymooAdapter(PymooProblem):
    """pymoo problem over the craft genome (continuous relaxation, rounded).

    pymoo minimizes, so the objective vector of the craft problem head is
    negated on the way in; results must be negated back on the way out. The
    continuous genes respect the integer bounds of the genome layout and are
    rounded to integers before every objective evaluation.
    """

    def __init__(self, problem, objectives) -> None:
        super().__init__(
            n_var=problem.genome_size,
            n_obj=2,
            xl=np.asarray(problem.lower_bounds, dtype=float),
            xu=np.asarray(problem.upper_bounds, dtype=float),
        )
        self.craft = problem
        self.objectives = objectives

    def _evaluate(self, x, out, *args, **kwargs) -> None:
        genomes = np.round(np.atleast_2d(x)).astype(np.int64)
        out["F"] = -np.array([self.objectives(genome) for genome in genomes])

    def genomes(self, x) -> np.ndarray:
        """Deduplicated integer genomes of a pymoo solution matrix."""
        rounded = np.round(np.atleast_2d(x)).astype(np.int64)
        unique = {genome.tobytes(): genome for genome in rounded}
        return np.array(list(unique.values()), dtype=np.int64)


def run_pymoo(seed, problem, objectives, epoch, pop_size) -> np.ndarray:
    """Run pymoo's NSGA-II; return its final front as integer genomes."""
    adapter = PymooAdapter(problem, objectives)
    result = minimize(
        adapter,
        PymooNSGA2(pop_size=pop_size),
        termination=("n_gen", epoch),
        seed=seed,
        verbose=False,
    )
    return adapter.genomes(result.X)


def measure_front(problem, evaluator: SolutionEvaluator, genomes) -> pd.DataFrame:
    """Measured front rows (penalized revenue, fairness) for the genomes."""
    rows = []
    for genome in genomes:
        solution = problem.decode(genome)
        point = evaluator.evaluate(solution.real, solution.discrete)
        rows.append({column: point[column] for column in FRONT_COLUMNS})
    return pd.DataFrame(rows)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    seeds = [int(seed) for seed in args.seeds.split(",")]

    problem = load_problem(args.timetable, args.revenue)
    evaluator = SolutionEvaluator(problem)
    objectives = revenue_fairness_objectives(problem, index=args.fair_index)

    tag = f"{args.timetable.parent.name}_{args.timetable.stem}"
    output_dir = args.output_dir / tag
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for seed in seeds:
        craft_solver = NSGA2(problem, pop_size=args.pop_size, objectives=objectives)
        craft_solver.optimize(epoch=args.epoch, seed=seed, verbose=False)

        pymoo_genomes = run_pymoo(seed, problem, objectives, args.epoch, args.pop_size)

        craft_front = measure_front(
            problem, evaluator, craft_solver.pareto_genomes
        )
        pymoo_front = measure_front(problem, evaluator, pymoo_genomes)
        craft_front.to_csv(output_dir / f"front_craft_seed{seed}.csv", index=False)
        pymoo_front.to_csv(output_dir / f"front_pymoo_seed{seed}.csv", index=False)

        values = {
            "craft": craft_front[FRONT_COLUMNS].to_numpy(),
            "pymoo": pymoo_front[FRONT_COLUMNS].to_numpy(),
        }
        union = np.vstack([values["craft"], values["pymoo"]])
        low, high = nadir(union), ideal(union)
        union_normalized = min_max_normalize(union, low, high)
        normalized = {
            name: min_max_normalize(points, low, high) for name, points in values.items()
        }
        row = {
            "seed": seed,
            "n_craft": len(craft_front),
            "n_pymoo": len(pymoo_front),
            "c_craft_over_pymoo": c_metric(values["craft"], values["pymoo"]),
            "c_pymoo_over_craft": c_metric(values["pymoo"], values["craft"]),
            "hypervolume_craft": normalized_hypervolume(
                normalized["craft"], normalized["pymoo"]
            ),
            "hypervolume_pymoo": normalized_hypervolume(
                normalized["pymoo"], normalized["craft"]
            ),
            "igd_craft": igd(normalized["craft"], union_normalized),
            "igd_pymoo": igd(normalized["pymoo"], union_normalized),
        }
        rows.append(row)
        if not args.quiet:
            print(
                f"seed {seed} | craft {row['n_craft']} pts "
                f"(HV {row['hypervolume_craft']:.3f}, IGD {row['igd_craft']:.3f}) | "
                f"pymoo {row['n_pymoo']} pts "
                f"(HV {row['hypervolume_pymoo']:.3f}, IGD {row['igd_pymoo']:.3f}) | "
                f"C(craft,pymoo)={row['c_craft_over_pymoo']:.2f} "
                f"C(pymoo,craft)={row['c_pymoo_over_craft']:.2f}"
            )

    results = pd.DataFrame(rows)
    results_path = output_dir / "pymoo_crosscheck.csv"
    results.to_csv(results_path, index=False)
    print(f"\nCross-check -> {results_path}")


if __name__ == "__main__":
    main()
