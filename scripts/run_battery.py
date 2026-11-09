#!/usr/bin/env python
"""Run the E1-E4 battery over the instance family manifest.

Reads the family manifest (``data/synthetic/instances/manifest.yaml``) and,
for every selected instance, drives the three validation harnesses as
subprocesses:

* ``comparison`` — the method x seed grid of ``scripts/run_comparison.py``
  (all methods, the rev*fair scalar sweep of the second paper via
  ``--fair-indices``), artifacts under ``--output-root/<instance tag>/``;
* ``epsilon`` — the exact epsilon-constraint reference of
  ``scripts/run_epsilon.py``, only for 25-service instances (the plan
  documents the exact model's intratability beyond that size);
* ``pymoo`` — the NSGA-II cross-check of ``scripts/run_pymoo.py`` (requires
  the optional ``baselines`` dependency group).

Every step skips instances whose artifact already exists, so the battery is
resumable; ``--force`` re-runs. A failed subprocess is reported at the end
and makes the script exit nonzero.

Examples:
    python scripts/run_battery.py --steps comparison --seeds 7,21,42,63,84
    python scripts/run_battery.py --instances n25_r3,n50_r3
    python scripts/run_battery.py --seeds "$(seq -s, 1 25)" --force
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

DEFAULT_SEEDS = "7,21,42,63,84"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the paper-3 battery over the instance family manifest."
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/synthetic/instances/manifest.yaml"),
        help="Instance family manifest (default: data/synthetic/instances/manifest.yaml)",
    )
    parser.add_argument(
        "--instances",
        type=str,
        default="",
        help="Comma-separated instance tags to run (default: all in the manifest)",
    )
    parser.add_argument(
        "--steps",
        type=str,
        default="comparison,epsilon,pymoo",
        help=(
            "Comma-separated battery steps (default: comparison,epsilon,pymoo; "
            "epsilon only applies to 25-service instances)"
        ),
    )
    parser.add_argument(
        "--seeds",
        type=str,
        default=DEFAULT_SEEDS,
        help=f"Comma-separated random seeds (default: {DEFAULT_SEEDS})",
    )
    parser.add_argument(
        "--fair-indices",
        type=str,
        default="jain,gini,atkinson",
        help="Fairness indices of the rev*fair scalar sweep (default: all three)",
    )
    parser.add_argument(
        "--pop-size", type=int, default=40, help="Population size (default: 40)"
    )
    parser.add_argument(
        "--epoch", type=int, default=100, help="Generations/iterations (default: 100)"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/results/battery"),
        help="Directory for the comparison artifacts (default: data/results/battery)",
    )
    parser.add_argument(
        "--epsilon-dir",
        type=Path,
        default=Path("data/results/epsilon"),
        help="Directory for the exact reference points (default: data/results/epsilon)",
    )
    parser.add_argument(
        "--pymoo-dir",
        type=Path,
        default=Path("data/results/pymoo"),
        help="Directory for the cross-check artifacts (default: data/results/pymoo)",
    )
    parser.add_argument(
        "--time-limit",
        type=float,
        default=300.0,
        help="SCIP time limit per epsilon solve, seconds (default: 300)",
    )
    parser.add_argument("--force", action="store_true", help="Re-run existing steps")
    parser.add_argument("--quiet", action="store_true", help="Silence child output")
    return parser


def run_step(script: str, arguments: list[str], quiet: bool) -> bool:
    """Run one battery step as a subprocess; report success."""
    print(f"  -> python scripts/{script} {' '.join(map(str, arguments))}")
    result = subprocess.run(
        [sys.executable, f"scripts/{script}", *map(str, arguments)],
        capture_output=quiet,
        text=True,
    )
    if result.returncode != 0:
        print(f"  FAILED (exit {result.returncode})")
        if quiet:
            print(result.stderr[-2000:])
        return False
    return True


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    steps = {step.strip() for step in args.steps.split(",")}
    selected = {tag.strip() for tag in args.instances.split(",") if tag.strip()}

    manifest = yaml.safe_load(args.manifest.read_text())
    instances = [
        entry
        for entry in manifest["instances"]
        if not selected or entry["tag"] in selected
    ]
    if not instances:
        parser.error("no instance selected (check --instances against the manifest)")
    print(
        f"Bateria E1-E4: {len(instances)} instancias, semillas {args.seeds}, "
        f"pasos {sorted(steps)}"
    )

    failures = []
    for entry in instances:
        tag = entry["tag"]
        print(f"\n=== {tag} ({entry['n_services']} servicios) ===")
        timetable = args.manifest.parent / entry["timetable"]
        revenue = args.manifest.parent / entry["revenue"]

        if "comparison" in steps:
            done = (args.output_root / f"{tag}_timetable" / "runs.csv").exists()
            if done and not args.force:
                print("  comparison: ya existe (usa --force para repetir)")
            else:
                ok = run_step(
                    "run_comparison.py",
                    [
                        "--timetable", timetable,
                        "--revenue", revenue,
                        "--methods", "all",
                        "--seeds", args.seeds,
                        "--fair-indices", args.fair_indices,
                        "--pop-size", args.pop_size,
                        "--epoch", args.epoch,
                        "--output-dir", args.output_root,
                        *(["--quiet"] if args.quiet else []),
                    ],
                    args.quiet,
                )
                if not ok:
                    failures.append(f"{tag}: comparison")

        if "epsilon" in steps and entry["n_services"] == 25:
            done = (
                args.epsilon_dir / f"{tag}_timetable" / "epsilon_reference.csv"
            ).exists()
            if done and not args.force:
                print("  epsilon: ya existe (usa --force para repetir)")
            else:
                ok = run_step(
                    "run_epsilon.py",
                    [
                        "--timetable", timetable,
                        "--revenue", revenue,
                        "--time-limit", args.time_limit,
                        "--output-dir", args.epsilon_dir,
                    ],
                    args.quiet,
                )
                if not ok:
                    failures.append(f"{tag}: epsilon")
        elif "epsilon" in steps:
            print("  epsilon: omitido (modelo exacto solo en 25 servicios)")

        if "pymoo" in steps:
            done = (
                args.pymoo_dir / f"{tag}_timetable" / "pymoo_crosscheck.csv"
            ).exists()
            if done and not args.force:
                print("  pymoo: ya existe (usa --force para repetir)")
            else:
                ok = run_step(
                    "run_pymoo.py",
                    [
                        "--timetable", timetable,
                        "--revenue", revenue,
                        "--seeds", args.seeds,
                        "--pop-size", args.pop_size,
                        "--epoch", args.epoch,
                        "--output-dir", args.pymoo_dir,
                        *(["--quiet"] if args.quiet else []),
                    ],
                    args.quiet,
                )
                if not ok:
                    failures.append(f"{tag}: pymoo")

    if failures:
        print(f"\nBateria terminada con fallos: {', '.join(failures)}")
        sys.exit(1)
    print(f"\nBateria completada ({len(instances)} instancias)")


if __name__ == "__main__":
    main()
