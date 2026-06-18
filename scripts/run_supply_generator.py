#!/usr/bin/env python
"""Generate a railway supply YAML file.

Uses the robin ``SupplyGenerator`` with the CRAFT supply/generator configs.
The output YAML can be fed to the optimization scripts (run_gsa, run_mealpy,
run_fairness) or to the robin KernelPlotter for visualization.

Examples:
    python scripts/run_supply_generator.py --n-services 25 --seed 42
    python scripts/run_supply_generator.py -o data/results/supply_custom.yaml
"""

import argparse
from pathlib import Path

from craft.runner import ExperimentConfig, ExperimentRunner


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a railway supply YAML.")
    parser.add_argument(
        "-n",
        "--n-services",
        type=int,
        default=25,
        help="Number of services to generate (default: 25)",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output YAML path (default: data/results/supply_generator_seed<seed>.yaml)",
    )
    parser.add_argument(
        "--supply-config",
        type=Path,
        default=Path("configs/supply_generator/supply_data.yaml"),
        help="Supply config YAML",
    )
    parser.add_argument(
        "--generator-config",
        type=Path,
        default=Path("configs/supply_generator/config.yaml"),
        help="Generator config YAML",
    )
    parser.add_argument(
        "--with-conflicts",
        action="store_true",
        help="Allow conflicts between services (default: without conflicts)",
    )
    args = parser.parse_args()

    output_path = args.output or Path(
        f"data/results/supply_generator_seed{args.seed}.yaml"
    )

    config = ExperimentConfig(
        supply_config_path=args.supply_config,
        generator_config_path=args.generator_config,
        output_supply_path=output_path,
        n_services=args.n_services,
        seed=args.seed,
        without_conflicts=not args.with_conflicts,
        algorithm="generator",
    )
    config.verbose = True

    runner = ExperimentRunner(config, project_root=Path.cwd())
    supply_path = runner.generate_supply()
    print(f"\nDone. Supply saved to: {supply_path}")


if __name__ == "__main__":
    main()
