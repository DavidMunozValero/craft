#!/usr/bin/env python
"""Run a fairness-aware timetabling optimization.

Generates a supply (or loads an existing one), builds the revenue-maximizing
timetabling problem, and solves it with a GA optimizing a weighted
combination of revenue and Jain's fairness index. The weights ``alpha``
(revenue) and ``beta`` (fairness) control the revenue-fairness trade-off.

Examples:
    python scripts/run_fairness.py --seed 42 --alpha 0.7 --beta 0.3
    python scripts/run_fairness.py --supply data/results/supply_custom.yaml --beta 0.5
    python scripts/run_fairness.py --epoch 100 --pop-size 30
"""

import argparse
from pathlib import Path

from craft.runner import ExperimentConfig, ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Run fairness-aware timetabling optimization.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument("--pop-size", type=int, default=20, help="Population size (default: 20)")
    parser.add_argument("--epoch", type=int, default=50, help="Number of epochs (default: 50)")
    parser.add_argument("--n-services", type=int, default=25, help="Number of services to generate (default: 25)")
    parser.add_argument("--alpha", type=float, default=0.7, help="Revenue weight (default: 0.7)")
    parser.add_argument("--beta", type=float, default=0.3, help="Fairness weight (default: 0.3)")
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
        algorithm="fairness",
        seed=args.seed,
        pop_size=args.pop_size,
        epoch=args.epoch,
        fairness_alpha=args.alpha,
        fairness_beta=args.beta,
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

    print(f"\nFairness results: fitness={results['fitness']:.2f}, scheduled={results['n_scheduled']}/{results['n_total']}")


if __name__ == "__main__":
    main()
