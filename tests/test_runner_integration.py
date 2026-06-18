"""Tests for craft.runner: ExperimentRunner pipeline (integration).

Runs the runner end-to-end with tiny parameters on a temp directory, verifying
that the full pipeline produces artifacts.
"""

from pathlib import Path

import pytest

from craft.runner import ExperimentConfig, ExperimentRunner


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class TestExperimentRunnerIntegration:
    @pytest.mark.parametrize("algorithm", ["ga", "de", "pso", "sca", "gsa"])
    def test_run_full_pipeline(self, algorithm, tmp_path):
        cfg = ExperimentConfig(
            n_services=6,
            algorithm=algorithm,
            seed=42,
            pop_size=10,
            epoch=3,
            verbose=False,
            results_dir=tmp_path / "results",
        )
        runner = ExperimentRunner(cfg, project_root=PROJECT_ROOT)
        results = runner.run()

        assert results["fitness"] > float("-inf")
        assert results["n_scheduled"] > 0
        assert results["n_total"] == 6
        assert results["saved"]["supply_path"].exists()
        assert results["saved"]["convergence_path"].exists()

    def test_load_existing_supply(self, supply_path, tmp_path):
        cfg = ExperimentConfig(
            algorithm="ga",
            seed=42,
            pop_size=10,
            epoch=3,
            verbose=False,
            results_dir=tmp_path / "results",
        )
        runner = ExperimentRunner(cfg, project_root=PROJECT_ROOT)
        runner.load_supply(supply_path)
        runner.build_timetabling()
        results = runner.optimize()
        results["saved"] = runner.save_results(results)

        assert results["fitness"] > 0
        assert results["saved"]["supply_path"].exists()
