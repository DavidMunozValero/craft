"""CRAFT — Capacity & Revenue-Aware Fair Timetabling.

Metaheuristic optimization of railway service timetables on top of the
`robin <https://github.com/JoseAngelMartinB/robin>`_ railway simulator.

Optimization backends are provided:
  * :class:`craft.gsa.GSA` — a custom hybrid Gravitational Search Algorithm.
  * :class:`craft.mealpy.MealpyTimetabling` — a problem formulation to be
    solved with the algorithms from the `mealpy` library.
  * :class:`craft.ga.GA` — a genetic algorithm over a direct genome
    (planning bits + minute deltas), formulated by
    :class:`craft.ga.GATimetabling`.

All build on the shared layer (entities, revenue, scheduling and fairness).
"""

from .common import Boundaries, ConflictMatrix, Solution
from .fairness import FairnessMetrics
from .revenue import RevenueCalculator, RevenueSimulator, penalty_function
from .scheduling import ScheduleManager, get_schedule_from_supply

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "Boundaries",
    "Solution",
    "ConflictMatrix",
    "FairnessMetrics",
    "RevenueSimulator",
    "RevenueCalculator",
    "penalty_function",
    "ScheduleManager",
    "get_schedule_from_supply",
    "GSA",
    "MealpyTimetabling",
    "GA",
    "GATimetabling",
]


def __getattr__(name: str):
    """Lazily import the optimization backends (heavy optional deps).

    ``GSA``, ``MealpyTimetabling``, ``GA`` and ``GATimetabling`` are resolved
    on first access so that ``import craft`` stays light and does not force
    importing scipy.spatial, the mealpy library or the GA backend unless a
    backend is actually used.
    """
    if name == "GSA":
        from .gsa import GSA

        return GSA
    if name == "MealpyTimetabling":
        from .mealpy import MealpyTimetabling

        return MealpyTimetabling
    if name in {"GA", "GATimetabling"}:
        from . import ga

        return getattr(ga, name)
    raise AttributeError(f"module 'craft' has no attribute {name!r}")


def __dir__() -> list:
    return sorted(
        list(globals().keys()) + ["GSA", "MealpyTimetabling", "GA", "GATimetabling"]
    )
