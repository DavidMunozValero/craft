"""Shared pytest fixtures for CRAFT tests.

Generates a small, reproducible supply YAML (8 services, seed=42) that can be
reused across the test modules. The supply is generated once per session and
cached on disk under a temporary path.
"""

from pathlib import Path

import pytest

from robin.supply.entities import Supply
from robin.supply.generator.entities import SupplyGenerator

from craft import RevenueSimulator
from craft.mealpy import MealpyTimetabling


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUPPLY_CONFIG = PROJECT_ROOT / "configs" / "supply_generator" / "supply_data.yaml"
GENERATOR_CONFIG = PROJECT_ROOT / "configs" / "supply_generator" / "config.yaml"


@pytest.fixture(scope="session")
def supply_path(tmp_path_factory) -> Path:
    """Generate a small supply YAML (8 services, seed=42) once per session."""
    tmp = tmp_path_factory.mktemp("supply")
    path = tmp / "supply_test.yaml"
    gen = SupplyGenerator.from_yaml(
        path_config_supply=SUPPLY_CONFIG,
        path_config_generator=GENERATOR_CONFIG,
    )
    gen.generate(
        n_services=8,
        output_path=path,
        seed=42,
        progress_bar=False,
        without_conflicts=False,
    )
    return path


@pytest.fixture(scope="session")
def supply(supply_path) -> Supply:
    """Load the generated supply."""
    return Supply.from_yaml(path=str(supply_path))


@pytest.fixture(scope="session")
def revenue_behavior(supply):
    """Compute revenue behavior for the supply (seeded for reproducibility)."""
    import numpy as np
    np.random.seed(42)
    return RevenueSimulator(supply=supply).simulate_revenue(alpha=2 / 3)


@pytest.fixture(scope="session")
def timetabling(supply, revenue_behavior) -> MealpyTimetabling:
    """Build the timetabling problem from the supply and revenue behavior."""
    return MealpyTimetabling(
        requested_services=supply.services,
        revenue_behavior=revenue_behavior,
        safe_headway=10,
        max_stop_time=10,
    )
