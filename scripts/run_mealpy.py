#!/usr/bin/env python
"""Run a mealpy-based timetabling optimization.

Generates a supply (or loads an existing one), builds the revenue-maximizing
timetabling problem, and solves it with one of the mealpy algorithms:
GA, DE, PSO, or SCA. The discrete part (scheduled services) is obtained with
the conflict-avoiding heuristic.

Examples:
    python scripts/run_mealpy.py --algorithm ga --seed 42 --epoch 50
    python scripts/run_mealpy.py -a de --supply data/results/supply_custom.yaml
    python scripts/run_mealpy.py -a pso --pop-size 30 --epoch 100
"""

import argparse
from pathlib import Path

from craft.runner import ExperimentConfig, ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run mealpy timetabling optimization.")
    parser.add_argument("-a", "--algorithm", choices=["ga", "de", "pso", "sca"], default="ga", help="mealpy algorithm (default: ga)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (default: 20)")
    parser.add_argument("--epoch", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--n-services", type=int, default=25, help="Number of services to generate (default: 25)")
    parser.add_argument("--supply", type=Path, default=None, help="Load an existing supply YAML instead of generating one")
    parser.add_argument("-o", "--output", type=Path, default=None, help="Output supply YAML path for generation (default: auto)")
    parser.add_argument("--supply-config", type=Path, default=Path("configs/supply_generator/supply_data.yaml"), help="Supply config YAML")
    parser.add_argument("--generator-config", type=Path, default=Path("configs/supply_generator/config.yaml"), help="Generator config YAML")
    parser.add_argument("--with-conflicts", action="store_true", help="Generate services with conflicts")
    parser.add_argument("--quiet", action="store_true", help="Suppress progress output")
    args = parser.parse_args()

    config = ExperimentConfig(
        supply_config_path=args.supply_config,
        generator_config_path=args.generator_config,
        output_supply_path=args.output,
        n_services=args.n_services,
        without_conflicts=not args.with_conflicts,
        algorithm=args.algorithm,
        seed=args.seed,
        pop_size=args.pop_size,
        epoch=args.epoch,
        verbose=not args.quiet,
    )

    runner = ExperimentRunner(config, project_root=Path.cwd())

    if args.supply:
        runner.load_supply(args.supply)
        runner.build_timetabling()
        results = runner.optimize()
        results["saved"] = runner.save_results(results)
    else:
        results = runner.run()

    print(f"\n{args.algorithm.upper()} results: fitness={results['fitness']:.2f}, scheduled={results['n_scheduled']}/{results['n_total']}")


if __name__ == "__main__":
    main()
