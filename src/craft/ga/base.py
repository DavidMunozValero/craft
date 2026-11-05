"""Integration contract between timetabling problems and solvers.

Two protocol levels define the single seam to plug new resolution methods
into the framework:

* :class:`Problem` — the minimal contract any solver needs (genome geometry
  plus the ``fitness``/``decode`` callbacks). Implement it to connect a new
  problem head to a solver, or consume it to connect a new solver (e.g. a
  custom GA) to any problem head::

      problem: Problem = load_problem(...)
      solver = MySolver(problem, ...)
      history = solver.optimize(epoch=..., seed=...)
      solution = problem.decode(solver.best_genome)

* :class:`DiagnosableProblem` — a ``Problem`` that additionally exposes the
  timetabling state and diagnostics consumed by the analysis and plotting
  helpers (:mod:`craft.ga.plotting`) and by the built-in GA history.
"""

from typing import (
    Any,
    Dict,
    List,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
    runtime_checkable,
)

import numpy as np

from ..common import Solution


@runtime_checkable
class Problem(Protocol):
    """Contract every problem head exposes to solvers.

    Attributes:
        genome_size: Total number of genes of an individual.
        lower_bounds: Inclusive lower bound per gene.
        upper_bounds: Inclusive upper bound per gene.
        n_services: Number of services (used by block-level operators).
        service_ids: Service ids in genome-block order.
        slices: Genome slice of every service block (``[b, δ…]``), so
            structure-aware operators can recombine whole blocks.
        time_slices: Slice of the delta genes of every service block.
    """

    genome_size: int
    lower_bounds: np.ndarray
    upper_bounds: np.ndarray
    n_services: int
    service_ids: Sequence[str]
    slices: List[slice]
    time_slices: List[slice]

    def fitness(self, genome: np.ndarray) -> float:
        """Penalized objective value of a genome (maximization)."""
        ...

    def decode(self, genome: np.ndarray) -> Solution:
        """Decode a genome into a candidate solution."""
        ...


@runtime_checkable
class DiagnosableProblem(Problem, Protocol):
    """A :class:`Problem` that also exposes diagnostics and state.

    Attributes:
        requested_schedules: Pristine requested timetable per service and
            station (``[arrival, departure]`` in minutes); never mutated.
        revenue_behavior: Revenue parameters per service.
        row_index: Matrix row of each service.
        services: Requested service objects.
        conflict_checkers: One conflict model per corridor.
    """

    requested_schedules: Mapping[str, Mapping[str, Sequence[float]]]
    revenue_behavior: Mapping[str, Mapping[str, float]]
    row_index: Mapping[str, int]
    services: Sequence[Any]
    conflict_checkers: Sequence[Any]

    def decoded_schedule(self, genome: np.ndarray) -> Dict[str, Dict[str, List[float]]]:
        """Complete decoded timetable (arrival/departure per service/station)."""
        ...

    def conflicting_pairs(self, genome: np.ndarray) -> List[Tuple[str, str]]:
        """Pairs of planned services in conflict after decoding."""
        ...
