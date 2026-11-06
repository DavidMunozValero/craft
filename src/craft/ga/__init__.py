"""Genetic algorithm backend with a direct (mask + deltas) genome.

Formulation of the revenue-maximization timetabling problem in which each
individual explicitly encodes which services are scheduled and their minute
deltas with respect to the requested timetables.
"""

from .algorithm import GA
from .nsga2 import NSGA2
from .timetabling import GATimetabling, load_problem

__all__ = ["GA", "GATimetabling", "NSGA2", "load_problem"]
