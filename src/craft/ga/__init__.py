"""Genetic algorithm backend with a direct (mask + deltas) genome.

Formulation of the revenue-maximization timetabling problem in which each
individual explicitly encodes which services are scheduled and their minute
deltas with respect to the requested timetables.
"""

from .algorithm import GA
from .evaluation import SolutionEvaluator, inequity_percent
from .indicators import (
    c_metric,
    ideal,
    igd,
    nadir,
    min_max_normalize,
    normalized_hypervolume,
    spread,
)
from .nsga2 import NSGA2
from .objectives import compute_capacities, revenue_fairness_objectives
from .timetabling import GATimetabling, load_problem

__all__ = [
    "GA",
    "GATimetabling",
    "NSGA2",
    "SolutionEvaluator",
    "c_metric",
    "compute_capacities",
    "ideal",
    "igd",
    "inequity_percent",
    "load_problem",
    "min_max_normalize",
    "nadir",
    "normalized_hypervolume",
    "revenue_fairness_objectives",
    "spread",
]
