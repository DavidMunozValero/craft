"""Tests for craft.runner: ExperimentConfig, build_mealpy_algorithm."""

from pathlib import Path

import pytest

from craft.runner import (
    DEFAULT_GENERATOR_CONFIG,
    DEFAULT_SUPPLY_CONFIG,
    ExperimentConfig,
    build_mealpy_algorithm,
)


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
