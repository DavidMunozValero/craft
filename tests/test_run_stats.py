"""End-to-end test for the statistical comparison script (``run_stats.py``)."""

from pathlib import Path

import pytest


def test_run_stats_reports_significant_pair(tmp_path):
    """A fully separated pair is flagged significant with a large effect."""
    import subprocess
    import sys

    import pandas as pd

    instance_dir = tmp_path / "inst1"
    instance_dir.mkdir()
    seeds = [1, 2, 3, 4, 5, 6]
    pd.DataFrame(
        {
            "method": ["m1"] * 6 + ["m2"] * 6,
            "seed": seeds * 2,
            "penalized": list(range(1, 7)) + [value + 10 for value in range(1, 7)],
        }
    ).to_csv(instance_dir / "runs.csv", index=False)

    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_stats.py"),
            "--benchmark-root",
            str(tmp_path),
            "--metrics",
            "penalized",
            "--alpha",
            "0.05",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=300,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    stats = pd.read_csv(instance_dir / "stats.csv")
    assert len(stats) == 1
    row = stats.iloc[0]
    assert row["method_a"] == "m1"
    assert row["method_b"] == "m2"
    assert row["cliffs_delta"] == pytest.approx(-1.0)
    assert row["magnitude"] == "large"
    assert bool(row["significant"])
    assert row["p_holm"] == pytest.approx(row["p_value"])
