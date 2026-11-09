#!/usr/bin/env python
"""Pairwise statistical comparison of battery runs (experiment E3).

For every instance directory under the benchmark root that holds a
``runs.csv``, compares every ordered method pair on the requested metrics
with the tools of :mod:`craft.stats`: the Wilcoxon signed-rank test on the
seed-paired samples, the Cliff's delta effect size and the Holm step-down
correction across the pairs of each (instance, metric) family. Writes
``stats.csv`` next to each ``runs.csv`` and prints the significant results
at the corrected level.

Examples:
    python scripts/run_stats.py
    python scripts/run_stats.py --benchmark-root data/results/battery --metrics penalized
"""

import argparse
from itertools import combinations
from pathlib import Path

import pandas as pd

from craft.stats import cliffs_delta, effect_magnitude, holm_correction, wilcoxon_signed_rank

DEFAULT_METRICS = "penalized,planned,conflicts,fairness_jain,inequity"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pairwise statistical comparison of the battery runs (E3)."
    )
    parser.add_argument(
        "--benchmark-root",
        type=Path,
        default=Path("data/results/battery"),
        help="Directory holding one subdirectory per instance (default: data/results/battery)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        default=DEFAULT_METRICS,
        help=f"Comma-separated metrics to compare (default: {DEFAULT_METRICS})",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for the Holm-corrected p-values (default: 0.05)",
    )
    return parser


def compare_instance(
    runs: pd.DataFrame, metrics: list[str], min_samples: int = 3
) -> pd.DataFrame:
    """Pairwise comparison table of one instance's runs.

    Args:
        runs: The ``runs.csv`` frame (one row per method and seed).
        metrics: Metrics to compare.
        min_samples: Minimum paired non-NaN samples per method; smaller
            pairs are skipped (n=1/2 cannot support the signed-rank test).

    Returns:
        One row per (metric, method pair) with descriptive statistics, the
        raw and Holm-adjusted p-values, the effect size and its magnitude.
    """
    methods = list(dict.fromkeys(runs["method"]))
    rows = []
    for metric in metrics:
        samples = {
            method: runs.loc[runs["method"] == method, metric].dropna()
            for method in methods
        }
        pairs = list(combinations(methods, 2))
        family = []
        for method_a, method_b in pairs:
            sample_a, sample_b = samples[method_a], samples[method_b]
            paired = min(len(sample_a), len(sample_b))
            if paired < min_samples:
                continue
            statistic, p_value = wilcoxon_signed_rank(sample_a, sample_b)
            family.append(
                {
                    "metric": metric,
                    "method_a": method_a,
                    "method_b": method_b,
                    "n": paired,
                    "mean_a": sample_a.mean(),
                    "std_a": sample_a.std(),
                    "mean_b": sample_b.mean(),
                    "std_b": sample_b.std(),
                    "statistic": statistic,
                    "p_value": p_value,
                    "cliffs_delta": cliffs_delta(sample_a, sample_b),
                }
            )
        if family:
            adjusted = holm_correction([row["p_value"] for row in family])
            for row, p_holm in zip(family, adjusted):
                row["p_holm"] = p_holm
                row["magnitude"] = effect_magnitude(row["cliffs_delta"])
            rows.extend(family)
    return pd.DataFrame(rows)


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    metrics = [metric.strip() for metric in args.metrics.split(",")]

    instance_dirs = sorted(
        path
        for path in args.benchmark_root.iterdir()
        if path.is_dir() and (path / "runs.csv").exists()
    )
    if not instance_dirs:
        parser.error(f"no runs.csv under {args.benchmark_root}")

    for instance_dir in instance_dirs:
        runs = pd.read_csv(instance_dir / "runs.csv")
        stats = compare_instance(runs, metrics)
        stats["significant"] = stats["p_holm"] < args.alpha
        stats_path = instance_dir / "stats.csv"
        stats.to_csv(stats_path, index=False)

        significant = stats[stats["significant"]]
        print(f"{instance_dir.name}: {len(stats)} pares -> {stats_path}")
        for row in significant.itertuples():
            print(
                f"  {row.metric:>14}: {row.method_a} vs {row.method_b} | "
                f"media {row.mean_a:.2f} vs {row.mean_b:.2f} | "
                f"p_holm={row.p_holm:.4f} | delta={row.cliffs_delta:+.2f} "
                f"({row.magnitude})"
            )
        if significant.empty:
            print("  (ninguna diferencia significativa tras Holm)")


if __name__ == "__main__":
    main()
