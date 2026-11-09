"""Tests for the pymoo cross-check (``scripts/run_pymoo.py``)."""

from pathlib import Path
from types import ModuleType

import importlib.util

import numpy as np
import pytest

pymoo = pytest.importorskip("pymoo")


from craft.ga import GATimetabling, revenue_fairness_objectives  # noqa: E402

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


def load_script(name: str) -> ModuleType:
    """Import a top-level script by file path."""
    path = Path(__file__).resolve().parent.parent / "scripts" / name
    spec = importlib.util.spec_from_file_location(name.removesuffix(".py"), path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture()
def problem(supply, revenue_behavior) -> GATimetabling:
    """Timetabling problem over the session fixture supply."""
    return GATimetabling(supply, revenue_behavior)


def test_adapter_negates_the_maximization_objectives(problem):
    """pymoo minimizes, so the wrapper objective is the negated vector."""
    module = load_script("run_pymoo.py")
    objectives = revenue_fairness_objectives(problem)
    adapter = module.PymooAdapter(problem, objectives)

    genome = np.zeros(problem.genome_size, dtype=np.int64)
    for block in problem.slices:
        genome[block.start] = 1
    batch = np.tile(genome.astype(float), (2, 1))
    batch[1, 0] += 0.4

    out = {}
    adapter._evaluate(batch, out)
    expected = -objectives(genome)
    assert out["F"].shape == (2, 2)
    assert out["F"][0] == pytest.approx(expected)
    assert out["F"][1] == pytest.approx(expected)


def test_adapter_rounds_and_deduplicates_genomes(problem):
    """Continuous genes are rounded to valid integer genomes, once each."""
    module = load_script("run_pymoo.py")
    adapter = module.PymooAdapter(problem, revenue_fairness_objectives(problem))

    genome = np.zeros(problem.genome_size, dtype=float)
    genome[0] = 0.6
    batch = np.vstack([genome, genome + np.eye(1, problem.genome_size, 0).ravel() * 0.0])
    result = adapter.genomes(batch)

    assert result.dtype == np.int64
    assert len(result) == 1
    assert result[0, 0] == 1
    assert (result[0] <= np.asarray(problem.upper_bounds)).all()
    assert (result[0] >= np.asarray(problem.lower_bounds)).all()


@pytest.mark.skipif(
    not (SYNTHETIC_DIR / "timetable.yaml").exists(),
    reason="synthetic data not generated (run notebook 0.0)",
)
def test_pymoo_crosscheck_script(tmp_path):
    """The cross-check script runs both solvers and writes the metrics."""
    import subprocess
    import sys

    project_root = Path(__file__).resolve().parent.parent
    result = subprocess.run(
        [
            sys.executable,
            str(project_root / "scripts" / "run_pymoo.py"),
            "--seeds",
            "42",
            "--pop-size",
            "10",
            "--epoch",
            "3",
            "--output-dir",
            str(tmp_path / "pymoo"),
            "--quiet",
        ],
        capture_output=True,
        text=True,
        cwd=project_root,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr[-2000:]

    out = tmp_path / "pymoo" / "synthetic_timetable"
    assert (out / "pymoo_crosscheck.csv").exists()
    assert (out / "front_craft_seed42.csv").exists()
    assert (out / "front_pymoo_seed42.csv").exists()

    import pandas as pd

    metrics = pd.read_csv(out / "pymoo_crosscheck.csv")
    assert metrics.loc[0, "n_craft"] > 0
    assert metrics.loc[0, "n_pymoo"] > 0
    for column in ("c_craft_over_pymoo", "c_pymoo_over_craft"):
        assert 0.0 <= metrics.loc[0, column] <= 1.0
