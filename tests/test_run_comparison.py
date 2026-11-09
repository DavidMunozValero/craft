"""End-to-end test for the benchmark runner script (``run_comparison.py``)."""

from pathlib import Path

import pytest

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "timetable.yaml").exists(),
    reason="synthetic data not generated (run notebook 0.0)",
)
def test_run_comparison_script(tmp_path):
    """The comparison script runs every method on a tiny budget and writes
    the structured benchmark artifacts."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_comparison.py"),
            "--methods",
            "all",
            "--seeds",
            "42",
            "--pop-size",
            "12",
            "--epoch",
            "5",
            "--output-dir",
            str(tmp_path / "benchmark"),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    out = tmp_path / "benchmark" / "synthetic_timetable"
    runs = out / "runs.csv"
    summary = out / "summary.csv"
    assert runs.exists()
    assert summary.exists()

    import pandas as pd

    df = pd.read_csv(runs)
    assert set(df["method"]) == {
        "greedy",
        "ga_mealpy",
        "ga_mealpy_fair_jain",
        "gsa",
        "ga_direct",
        "nsga2",
    }
    assert len(df) == 6
    assert (df["planned"] >= 0).all()
    assert (out / "pareto_front_nsga2_seed42.csv").exists()
    assert (out / "indicators_nsga2.csv").exists()
    assert (out / "convergence_gsa_seed42.csv").exists()
    assert (out / "best_schedule_greedy_seed42.csv").exists()
