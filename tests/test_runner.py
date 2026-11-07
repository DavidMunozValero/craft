"""Tests for craft.runner: ExperimentConfig, build_mealpy_algorithm."""

from pathlib import Path

import pytest

from craft.runner import (
    DEFAULT_GENERATOR_CONFIG,
    DEFAULT_SUPPLY_CONFIG,
    ExperimentConfig,
    ExperimentRunner,
    build_mealpy_algorithm,
)

SYNTHETIC_DIR = Path(__file__).resolve().parent.parent / "data" / "synthetic"


class TestExperimentConfig:
    def test_defaults(self):
        cfg = ExperimentConfig()
        assert cfg.algorithm == "ga"
        assert cfg.seed == 42
        assert cfg.pop_size == 20
        assert cfg.epoch == 50
        assert cfg.n_services == 25

    def test_custom_values(self):
        cfg = ExperimentConfig(algorithm="gsa", seed=99, pop_size=30, epoch=100)
        assert cfg.algorithm == "gsa"
        assert cfg.seed == 99
        assert cfg.pop_size == 30
        assert cfg.epoch == 100

    def test_resolved_paths(self, tmp_path):
        root = tmp_path / "project"
        root.mkdir()
        cfg = ExperimentConfig()
        paths = cfg.resolved_paths(root)
        assert paths["supply_config"] == root / DEFAULT_SUPPLY_CONFIG
        assert paths["generator_config"] == root / DEFAULT_GENERATOR_CONFIG
        assert paths["results_dir"] == root / "data/results"
        assert paths["output_supply"] is None

    def test_resolved_paths_absolute(self):
        cfg = ExperimentConfig(supply_config_path=Path("/absolute/path.yaml"))
        paths = cfg.resolved_paths(Path("/some/root"))
        assert paths["supply_config"] == Path("/absolute/path.yaml")


class TestBuildMealpyAlgorithm:
    @pytest.mark.parametrize("name", ["ga", "de", "pso", "sca"])
    def test_build_known_algorithms(self, name):
        model = build_mealpy_algorithm(name, epoch=10, pop_size=5)
        assert model is not None
        assert hasattr(model, "solve")

    def test_unknown_algorithm_raises(self):
        with pytest.raises(ValueError, match="Unknown algorithm"):
            build_mealpy_algorithm("xyz", epoch=10, pop_size=5)

    def test_case_insensitive(self):
        model = build_mealpy_algorithm("GA", epoch=10, pop_size=5)
        assert model is not None


class TestInstanceMode:
    def test_resolved_instance_paths(self, tmp_path):
        timetable = tmp_path / "timetable.yaml"
        revenue = tmp_path / "revenue.yaml"
        cfg = ExperimentConfig(timetable_path=timetable, revenue_path=revenue)
        paths = cfg.resolved_paths(tmp_path)
        assert paths["timetable"] == timetable
        assert paths["revenue"] == revenue

    def test_load_instance_requires_both_paths(self, tmp_path):
        cfg = ExperimentConfig(timetable_path=tmp_path / "timetable.yaml")
        runner = ExperimentRunner(cfg, project_root=tmp_path)
        with pytest.raises(ValueError):
            runner.load_instance()

    @pytest.mark.skipif(
        not (SYNTHETIC_DIR / "timetable.yaml").exists(),
        reason="synthetic data not generated (run notebook 0.0)",
    )
    def test_load_instance_builds_the_priced_problem(self, tmp_path):
        import yaml

        revenue_payload = yaml.safe_load((SYNTHETIC_DIR / "revenue.yaml").read_text())[
            "services"
        ]
        cfg = ExperimentConfig(
            timetable_path=SYNTHETIC_DIR / "timetable.yaml",
            revenue_path=SYNTHETIC_DIR / "revenue.yaml",
            verbose=False,
        )
        runner = ExperimentRunner(cfg, project_root=tmp_path)
        runner.load_instance()

        tt = runner.timetabling
        assert tt is not None
        assert tt.n_services == 50
        assert len(tt.boundaries.real) == sum(
            len(schedule) - 1 for schedule in tt.reference_schedules.values()
        )
        assert tt.revenue == revenue_payload
        expected_rus = {str(params["ru"]) for params in revenue_payload.values()}
        assert set(tt.capacities) == expected_rus
        assert runner.paths["output_supply"] == SYNTHETIC_DIR / "timetable.yaml"
