"""End-to-end test for the battery orchestrator (``run_battery.py``)."""

from pathlib import Path

import pytest

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "instances" / "manifest.yaml").exists(),
    reason="instance family not generated (run notebook 0.0, section 8)",
)
def test_run_battery_single_instance(tmp_path):
    """The orchestrator drives the comparison grid for one manifest entry."""
    import subprocess
    import sys

    import yaml

    instance_dir = SYNTHETIC_DIR / "instances" / "n25_r3"
    manifest = {
        "metadata": {"description": "battery test manifest"},
        "instances": [
            {
                "tag": "n25_r3",
                "scenario": "r3",
                "mode": "sampled",
                "n_services": 25,
                "seed": 42,
                "timetable": str(instance_dir / "timetable.yaml"),
                "revenue": str(instance_dir / "revenue.yaml"),
            }
        ],
    }
    manifest_path = tmp_path / "manifest.yaml"
    manifest_path.write_text(yaml.safe_dump(manifest))

    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_battery.py"),
            "--manifest",
            str(manifest_path),
            "--instances",
            "n25_r3",
            "--steps",
            "comparison",
            "--seeds",
            "42",
            "--pop-size",
            "12",
            "--epoch",
            "5",
            "--fair-indices",
            "jain",
            "--output-root",
            str(tmp_path / "battery"),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=900,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    runs = tmp_path / "battery" / "n25_r3_timetable" / "runs.csv"
    assert runs.exists()

    import pandas as pd

    df = pd.read_csv(runs)
    assert "ga_mealpy_fair_jain" in set(df["method"])
    assert (tmp_path / "battery" / "n25_r3_timetable" / "summary.csv").exists()
