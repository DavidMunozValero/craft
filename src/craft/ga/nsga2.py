"""NSGA-II for the direct timetabling encoding.

Self-contained implementation of the Non-dominated Sorting Genetic Algorithm
II (Deb et al., 2002) over the same genome and operators as the GA backend:
fast non-dominated sorting, crowding distance assignment, binary tournament
with the crowded comparison operator and mu plus lambda environmental
selection. With the default single objective (the penalized revenue of the
problem head) the algorithm degrades gracefully to an elitist rank-based GA,
while staying ready for genuinely multi-objective formulations.

Conflicts are not handled through constraint domination: they are already
priced into the penalized fitness, exactly as in the GA, so both solvers
optimize the same landscape and remain comparable.
"""

import time
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..common import Solution
from .base import DiagnosableProblem
from .operators import BlockOperators


def dominates(a: np.ndarray, b: np.ndarray) -> bool:
    """Whether objective vector ``a`` dominates ``b`` (maximization).

    Args:
        a: Objective vector of the candidate dominator.
        b: Objective vector of the candidate dominated.

    Returns:
        True if ``a`` is at least as good as ``b`` everywhere and strictly
        better somewhere.
    """
    return bool((a >= b).all() and (a > b).any())


def fast_non_dominated_sort(values: np.ndarray) -> List[np.ndarray]:
    """Peel the population into non-dominated fronts, best first.

    Args:
        values: Objective matrix of shape ``(n, n_objectives)``.

    Returns:
        List of index arrays, one per front in ascending domination order;
        the first front is the non-dominated set of the population.
    """
    dominates_matrix = np.logical_and(
        (values[:, None, :] >= values[None, :, :]).all(axis=2),
        (values[:, None, :] > values[None, :, :]).any(axis=2),
    )
    domination_count = dominates_matrix.sum(axis=0)
    remaining = np.ones(values.shape[0], dtype=bool)
    fronts: List[np.ndarray] = []
    while remaining.any():
        front = np.nonzero(remaining & (domination_count == 0))[0]
        fronts.append(front)
        remaining[front] = False
        domination_count -= dominates_matrix[front].sum(axis=0)
    return fronts


def crowding_distance(values: np.ndarray) -> np.ndarray:
    """Crowding distance of every individual within its front.

    Boundary solutions of each objective get an infinite distance so they
    are always preserved; interior solutions accumulate the normalized gap
    between their neighbours. Objectives with zero span are skipped.

    Args:
        values: Objective matrix of the front, shape ``(n, n_objectives)``.

    Returns:
        Crowding distance per individual, shape ``(n,)``.
    """
    n = values.shape[0]
    distance = np.zeros(n)
    if n <= 2:
        return np.full(n, np.inf)
    for objective in range(values.shape[1]):
        order = np.argsort(values[:, objective], kind="stable")
        distance[order[0]] = np.inf
        distance[order[-1]] = np.inf
        span = values[order[-1], objective] - values[order[0], objective]
        if span > 0:
            distance[order[1:-1]] += (
                values[order[2:], objective] - values[order[:-2], objective]
            ) / span
    return distance


def knee_index(front: np.ndarray) -> int:
    """Index of the front point closest to the ideal corner.

    Objectives are min-max normalized over the front and the knee is the
    point with the smallest Euclidean distance to the ideal (all-ones)
    corner.

    Args:
        front: Objective matrix of the Pareto front, shape ``(n, m)``.

    Returns:
        Row index of the knee point within ``front``.
    """
    low = front.min(axis=0)
    span = front.max(axis=0) - low
    normalized = (front - low) / np.where(span > 0, span, 1.0)
    return int(np.argmin(np.linalg.norm(normalized - 1.0, axis=1)))


def hypervolume(values: np.ndarray, reference_point: Sequence[float]) -> float:
    """Exact hypervolume of a two-objective front under maximization.

    Only the points weakly dominating the reference point contribute; the
    front is swept from the largest first objective to the smallest and the
    covered area is accumulated per band of the first objective.

    Args:
        values: Objective matrix of shape ``(n, 2)``.
        reference_point: Reference point dominated by the front of interest
            (e.g. the nadir); points not dominating it are ignored.

    Returns:
        The dominated area between the front and the reference point.

    Raises:
        ValueError: If ``values`` does not hold exactly two objectives.
    """
    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return 0.0
    if values.ndim != 2 or values.shape[1] != 2:
        raise ValueError("hypervolume only supports two objectives")
    ref = np.asarray(reference_point, dtype=float)
    pts = values[(values[:, 0] >= ref[0]) & (values[:, 1] >= ref[1])]
    if pts.shape[0] == 0:
        return 0.0
    order = np.argsort(pts[:, 0], kind="stable")
    area = 0.0
    ceiling = ref[1]
    for i in range(pts.shape[0] - 1, -1, -1):
        ceiling = max(ceiling, pts[order[i], 1])
        left = pts[order[i - 1], 0] if i > 0 else ref[0]
        area += (pts[order[i], 0] - left) * (ceiling - ref[1])
    return float(area)


class NSGA2:
    """NSGA-II solver for the revenue-maximization timetabling problem.

    Works over any object satisfying the
    :class:`~craft.ga.base.DiagnosableProblem` contract and reuses the
    block-aware operators of the GA backend, so both solvers explore the
    same encoding with the same variation mechanics and differ only in
    selection and replacement.

    Attributes:
        best_genome: Best genome found w.r.t. the first objective (empty
            until :meth:`optimize` runs).
        best_fitness: First objective of the best genome found.
        best_solution: Decoded :class:`~craft.common.Solution` of the best
            genome found.
        best_genomes: Best genome at the end of every generation, including
            the initial one (handy to animate or plot the evolution).
        convergence: Best fitness at the end of every generation, including
            the initial one (length ``epoch + 1``).
        pareto_genomes: Genomes of the final non-dominated front, sorted by
            descending first objective.
        pareto_values: Objective matrix of :attr:`pareto_genomes`.
        pareto_history: Objective matrix of the first front at the end of
            every generation, including the initial one (length
            ``epoch + 1``); the raw material for hypervolume curves.
        execution_time: Seconds spent inside :meth:`optimize`.
    """

    def __init__(
        self,
        problem: DiagnosableProblem,
        pop_size: int = 50,
        pc: float = 0.9,
        pm: float = 0.05,
        sigma: int = 2,
        pr: float = 0.1,
        objectives: Optional[Callable[[np.ndarray], np.ndarray]] = None,
    ) -> None:
        """Configure NSGA-II over a timetabling problem.

        Args:
            problem: Problem defining the genome layout and the fitness.
            pop_size: Population size (at least 10, kept aligned with the
                GA backend so shared hyperparameters stay comparable).
            pc: Crossover probability per pair of parents.
            pm: Per-gene mutation probability: planning bits flip with this
                probability and deltas are perturbed with it too.
            sigma: Standard deviation, in minutes, of the integer gaussian
                applied to mutated deltas.
            pr: Probability that a mutated delta resets to 0 (the time
                requested by the RU) instead of being perturbed.
            objectives: Map from genome to objective vector (maximization).
                Defaults to the penalized revenue of the problem head as a
                single objective, the very fitness optimized by the GA.
        """
        if pop_size < 10:
            raise ValueError("pop_size must be at least 10 (tournament selection)")
        self.problem = problem
        self.pop_size = pop_size
        self._ops = BlockOperators(problem, pc=pc, pm=pm, sigma=sigma, pr=pr)
        self._objectives = objectives or (
            lambda genome: np.array([problem.fitness(genome)])
        )

        self.best_genome: np.ndarray = np.empty(0, dtype=np.int64)
        self.best_fitness: float = float("-inf")
        self.best_solution: Solution = Solution(np.empty(0), np.empty(0))
        self.best_genomes: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self.convergence: np.ndarray = np.empty(0)
        self.pareto_genomes: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self.pareto_values: np.ndarray = np.empty((0, 0))
        self.pareto_history: List[np.ndarray] = []
        self.execution_time: float = 0.0

    def optimize(
        self, epoch: int = 100, seed: Optional[int] = None, verbose: bool = True
    ) -> pd.DataFrame:
        """Run the evolution and return its history.

        Args:
            epoch: Number of generations.
            seed: Seed for the random generator (reproducible runs).
            verbose: Whether to print per-generation progress.

        Returns:
            History DataFrame with one row per generation and the columns
            ``Iteration``, ``Fitness``, ``Planned``, ``Conflicts`` and
            ``ExecutionTime`` of the best individual.
        """
        rng = np.random.default_rng(seed)
        start = time.time()

        population = self._ops.initial_population(self.pop_size, rng)
        values = self._evaluate(population)
        self._track(population, values)

        convergence = np.empty(epoch + 1)
        snapshots = np.empty((epoch + 1, self.problem.genome_size), dtype=np.int64)
        convergence[0] = self.best_fitness
        snapshots[0] = self.best_genome
        pareto_history: List[np.ndarray] = [self._front(values)]
        records: List[Tuple] = [self._record(0, 0.0)]

        for generation in range(1, epoch + 1):
            offspring = self._breed(population, values, rng)
            merged = np.vstack((population, offspring))
            merged_values = np.vstack((values, self._evaluate(offspring)))
            population, values = self._select(merged, merged_values)
            self._track(population, values)
            convergence[generation] = self.best_fitness
            snapshots[generation] = self.best_genome
            pareto_history.append(self._front(values))
            elapsed = time.time() - start
            records.append(self._record(generation, elapsed))
            if verbose:
                print(
                    f"Generation {generation:3d} | best fitness "
                    f"{self.best_fitness:.2f} | planned "
                    f"{records[-1][2]} | conflicts {records[-1][3]}"
                )

        self.execution_time = time.time() - start
        self.convergence = convergence
        self.best_genomes = snapshots
        self.pareto_history = pareto_history
        self.best_solution = self.problem.decode(self.best_genome)
        front = fast_non_dominated_sort(values)[0]
        order = np.argsort(-values[front, 0], kind="stable")
        self.pareto_genomes = population[front][order]
        self.pareto_values = values[front][order]
        return pd.DataFrame(
            records,
            columns=["Iteration", "Fitness", "Planned", "Conflicts", "ExecutionTime"],
        )

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        """Objective matrix of every individual in the population."""
        values: np.ndarray = np.asarray(
            [self._objectives(genome) for genome in population], dtype=float
        )
        return values.reshape(len(population), -1)

    def _front(self, values: np.ndarray) -> np.ndarray:
        """Objective matrix of the first non-dominated front."""
        front = fast_non_dominated_sort(values)[0]
        return np.array(values[front], copy=True)

    def _breed(
        self,
        population: np.ndarray,
        values: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build the offspring via crowded binary tournaments."""
        fronts = fast_non_dominated_sort(values)
        rank = np.empty(len(population), dtype=int)
        distance = np.empty(len(population))
        for level, front in enumerate(fronts):
            rank[front] = level
            distance[front] = crowding_distance(values[front])
        offspring: List[np.ndarray] = []
        while len(offspring) < self.pop_size:
            p1 = self._tournament(population, rank, distance, rng)
            p2 = self._tournament(population, rank, distance, rng)
            for child in self._ops.crossover(p1, p2, rng):
                offspring.append(self._ops.mutate(child, rng))
        return np.array(offspring[: self.pop_size], dtype=np.int64)

    def _tournament(
        self,
        population: np.ndarray,
        rank: np.ndarray,
        distance: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the winner of a binary crowded-comparison tournament."""
        a, b = rng.integers(0, len(population), size=2)
        if rank[a] != rank[b]:
            winner = a if rank[a] < rank[b] else b
        else:
            winner = a if distance[a] >= distance[b] else b
        selected: np.ndarray = population[winner]
        return selected

    def _select(
        self, population: np.ndarray, values: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Mu plus lambda environmental selection.

        Survivors are taken front by front; the front crossing the
        population budget is truncated by descending crowding distance.
        """
        chosen: List[int] = []
        for front in fast_non_dominated_sort(values):
            if len(chosen) + len(front) <= self.pop_size:
                chosen.extend(front.tolist())
                continue
            slots = self.pop_size - len(chosen)
            distance = crowding_distance(values[front])
            order = np.argsort(-distance, kind="stable")
            chosen.extend(front[order[:slots]].tolist())
            break
        survivors = np.array(chosen, dtype=int)
        return population[survivors], values[survivors]

    def _track(self, population: np.ndarray, values: np.ndarray) -> None:
        """Update the best genome found so far (first objective)."""
        index = int(np.argmax(values[:, 0]))
        if values[index, 0] > self.best_fitness:
            self.best_fitness = float(values[index, 0])
            self.best_genome = population[index].copy()

    def _record(self, generation: int, elapsed: float) -> Tuple:
        """Summary tuple of the current best individual for the history."""
        conflicts = len(self.problem.conflicting_pairs(self.best_genome))
        planned = int(np.sum(self.problem.decode(self.best_genome).discrete))
        return generation, self.best_fitness, planned, conflicts, elapsed
