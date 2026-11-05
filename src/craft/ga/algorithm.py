"""Genetic algorithm for the direct timetabling encoding.

Block-structured GA over the genome layout of
:class:`craft.ga.timetabling.GATimetabling`: k-tournament selection, uniform
crossover at service-block granularity, integer gaussian mutation with
reset-to-requested-time events, and elitism. Every decoded genome is feasible
by construction, so the search only trades off revenue against the conflict
penalty.
"""

import time
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from ..common import Solution
from .base import DiagnosableProblem


class GA:
    """Genetic algorithm for the revenue-maximization timetabling problem.

    Works over any object satisfying the
    :class:`~craft.ga.base.DiagnosableProblem` contract (the built-in
    :class:`~craft.ga.timetabling.GATimetabling` or any other problem head),
    so alternative solvers can be plugged in without touching the
    formulation.

    Attributes:
        best_genome: Best genome found (empty until :meth:`optimize` runs).
        best_fitness: Fitness of the best genome found.
        best_solution: Decoded :class:`~craft.common.Solution` of the best
            genome found.
        best_genomes: Best genome at the end of every generation, including
            the initial one (handy to animate or plot the evolution).
        convergence: Best fitness at the end of every generation, including
            the initial one (length ``epoch + 1``).
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
        elite: int = 1,
        tournament_size: int = 3,
    ) -> None:
        """Configure the GA over a timetabling problem.

        Args:
            problem: Problem defining the genome layout and the fitness.
            pop_size: Population size (at least 10, as tournament selection
                requires enough candidates).
            pc: Crossover probability per pair of parents.
            pm: Per-gene mutation probability: planning bits flip with this
                probability and deltas are perturbed with it too.
            sigma: Standard deviation, in minutes, of the integer gaussian
                applied to mutated deltas.
            pr: Probability that a mutated delta resets to 0 (the time
                requested by the RU) instead of being perturbed.
            elite: Number of best individuals copied to the next generation.
            tournament_size: Candidates per tournament.
        """
        if pop_size < 10:
            raise ValueError("pop_size must be at least 10 (tournament selection)")
        self.problem = problem
        self.pop_size = pop_size
        self.pc = pc
        self.pm = pm
        self.sigma = sigma
        self.pr = pr
        self.elite = elite
        self.tournament_size = tournament_size

        self._starts = np.array([block.start for block in problem.slices])
        self._widths = np.array([block.stop - block.start for block in problem.slices])
        self._is_bit = np.zeros(problem.genome_size, dtype=bool)
        self._is_bit[self._starts] = True

        self.best_genome: np.ndarray = np.empty(0, dtype=np.int64)
        self.best_fitness: float = float("-inf")
        self.best_solution: Solution = Solution(np.empty(0), np.empty(0))
        self.best_genomes: np.ndarray = np.empty((0, 0), dtype=np.int64)
        self.convergence: np.ndarray = np.empty(0)
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

        population = self._initial_population(rng)
        fitness = self._evaluate(population)
        self._track(population, fitness)

        convergence = np.empty(epoch + 1)
        snapshots = np.empty((epoch + 1, self.problem.genome_size), dtype=np.int64)
        convergence[0] = self.best_fitness
        snapshots[0] = self.best_genome
        records: List[Tuple] = [self._record(0, 0.0)]

        for generation in range(1, epoch + 1):
            population = self._breed(population, fitness, rng)
            fitness = self._evaluate(population)
            self._track(population, fitness)
            convergence[generation] = self.best_fitness
            snapshots[generation] = self.best_genome
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
        self.best_solution = self.problem.decode(self.best_genome)
        return pd.DataFrame(
            records,
            columns=["Iteration", "Fitness", "Planned", "Conflicts", "ExecutionTime"],
        )

    def _initial_population(self, rng: np.random.Generator) -> np.ndarray:
        """Sample the initial population around the requested timetables.

        Every service starts planned (``b = 1``) and the deltas carry a small
        gaussian noise around 0, clipped to the genome bounds.
        """
        population = np.zeros((self.pop_size, self.problem.genome_size), dtype=np.int64)
        population += np.round(rng.normal(0.0, self.sigma, population.shape)).astype(
            np.int64
        )
        population[:, self._starts] = 1
        np.clip(
            population,
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            out=population,
        )
        return population

    def _evaluate(self, population: np.ndarray) -> np.ndarray:
        """Fitness of every individual in the population."""
        fitness: np.ndarray = np.array(
            [self.problem.fitness(genome) for genome in population], dtype=float
        )
        return fitness

    def _breed(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Build the next generation: elites plus bred offspring."""
        offspring = [population[i].copy() for i in np.argsort(fitness)[-self.elite :]]
        while len(offspring) < self.pop_size:
            p1 = self._tournament(population, fitness, rng)
            p2 = self._tournament(population, fitness, rng)
            for child in self._crossover(p1, p2, rng):
                offspring.append(self._mutate(child, rng))
        return np.array(offspring[: self.pop_size], dtype=np.int64)

    def _tournament(
        self,
        population: np.ndarray,
        fitness: np.ndarray,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """Return the fittest of ``tournament_size`` random individuals."""
        candidates = rng.integers(0, self.pop_size, size=self.tournament_size)
        selected: np.ndarray = population[
            int(candidates[int(np.argmax(fitness[candidates]))])
        ]
        return selected

    def _crossover(
        self, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recombine two parents exchanging whole service blocks.

        Each child takes every service block from one of the parents, so the
        planning bit of a service always travels together with its deltas.
        """
        if rng.random() >= self.pc:
            return p1.copy(), p2.copy()
        take = np.repeat(rng.random(self.problem.n_services) < 0.5, self._widths)
        return np.where(take, p2, p1), np.where(take, p1, p2)

    def _mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mutate a genome in place semantics: planning bits flip, deltas are
        reset to the requested time or perturbed with an integer gaussian.

        The result is clipped to the genome bounds, so every child remains a
        valid individual.
        """
        child = genome.copy()
        rand = rng.random(child.shape)
        child[(rand < self.pm) & self._is_bit] ^= 1
        noise = np.round(rng.normal(0.0, self.sigma, child.shape)).astype(np.int64)
        reset = (rand < self.pr) & ~self._is_bit
        perturb = (rand >= self.pr) & (rand < self.pr + self.pm) & ~self._is_bit
        child = np.where(reset, 0, child + np.where(perturb, noise, 0))
        np.clip(child, self.problem.lower_bounds, self.problem.upper_bounds, out=child)
        return child

    def _track(self, population: np.ndarray, fitness: np.ndarray) -> None:
        """Update the best genome found so far."""
        index = int(np.argmax(fitness))
        if fitness[index] > self.best_fitness:
            self.best_fitness = float(fitness[index])
            self.best_genome = population[index].copy()

    def _record(self, generation: int, elapsed: float) -> Tuple:
        """Summary tuple of the current best individual for the history."""
        conflicts = len(self.problem.conflicting_pairs(self.best_genome))
        planned = int(np.sum(self.problem.decode(self.best_genome).discrete))
        return generation, self.best_fitness, planned, conflicts, elapsed
