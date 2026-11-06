"""Block-aware genetic operators shared by the genetic backends.

The operators wrap the genome layout described by a
:class:`~craft.ga.base.Problem` (one planning bit plus minute deltas per
service block) so every genetic solver (GA, NSGA-II, ...) reuses the same
variation machinery: a seeded initial population, uniform crossover at
service-block granularity and integer gaussian mutation with reset events.
All operators keep children inside the genome bounds, so every individual
they produce is valid by construction.
"""

from typing import Tuple

import numpy as np

from .base import Problem


class BlockOperators:
    """Variation operators over the direct (mask + deltas) genome.

    Attributes:
        starts: First gene index of every service block.
        widths: Number of genes of every service block.
        is_bit: Mask marking the planning-bit gene of every block.
    """

    def __init__(
        self,
        problem: Problem,
        pc: float = 0.9,
        pm: float = 0.05,
        sigma: int = 2,
        pr: float = 0.1,
    ) -> None:
        """Bind the operators to a problem and its hyperparameters.

        Args:
            problem: Problem defining the genome layout and bounds.
            pc: Crossover probability per pair of parents.
            pm: Per-gene mutation probability.
            sigma: Standard deviation, in minutes, of the integer gaussian
                applied to mutated deltas.
            pr: Probability that a mutated delta resets to 0 (the time
                requested by the RU) instead of being perturbed.
        """
        self.problem = problem
        self.pc = pc
        self.pm = pm
        self.sigma = sigma
        self.pr = pr
        self.starts = np.array([block.start for block in problem.slices])
        self.widths = np.array([block.stop - block.start for block in problem.slices])
        self.is_bit = np.zeros(problem.genome_size, dtype=bool)
        self.is_bit[self.starts] = True

    def initial_population(self, pop_size: int, rng: np.random.Generator) -> np.ndarray:
        """Sample a population around the requested timetables.

        Every service starts planned (``b = 1``) and the deltas carry a
        small gaussian noise around 0, clipped to the genome bounds.

        Args:
            pop_size: Number of individuals to sample.
            rng: Random generator used for the sampling.

        Returns:
            Integer array of shape ``(pop_size, problem.genome_size)``.
        """
        population = np.zeros((pop_size, self.problem.genome_size), dtype=np.int64)
        population += np.round(rng.normal(0.0, self.sigma, population.shape)).astype(
            np.int64
        )
        population[:, self.starts] = 1
        np.clip(
            population,
            self.problem.lower_bounds,
            self.problem.upper_bounds,
            out=population,
        )
        return population

    def crossover(
        self, p1: np.ndarray, p2: np.ndarray, rng: np.random.Generator
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Recombine two parents exchanging whole service blocks.

        Each child takes every service block from one of the parents, so
        the planning bit of a service always travels together with its
        deltas.

        Args:
            p1: First parent genome.
            p2: Second parent genome.
            rng: Random generator used for the block exchange.

        Returns:
            The two children genomes.
        """
        if rng.random() >= self.pc:
            return p1.copy(), p2.copy()
        take = np.repeat(rng.random(self.problem.n_services) < 0.5, self.widths)
        return np.where(take, p2, p1), np.where(take, p1, p2)

    def mutate(self, genome: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Mutate a genome: planning bits flip, deltas are reset to the
        requested time or perturbed with an integer gaussian.

        The result is clipped to the genome bounds, so every child remains
        a valid individual.

        Args:
            genome: Genome to mutate (left untouched).
            rng: Random generator used for the mutation events.

        Returns:
            A new mutated genome.
        """
        child = genome.copy()
        rand = rng.random(child.shape)
        child[(rand < self.pm) & self.is_bit] ^= 1
        noise = np.round(rng.normal(0.0, self.sigma, child.shape)).astype(np.int64)
        reset = (rand < self.pr) & ~self.is_bit
        perturb = (rand >= self.pr) & (rand < self.pr + self.pm) & ~self.is_bit
        child = np.where(reset, 0, child + np.where(perturb, noise, 0))
        np.clip(child, self.problem.lower_bounds, self.problem.upper_bounds, out=child)
        return child
