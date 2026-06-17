"""Shared entities used across CRAFT modules.

This module centralizes the data structures that are reused by both the
GSA (custom) and mealpy-based optimization backends, so there is a single
source of truth for boundaries, solutions and the service conflict matrix.
"""

from typing import Any, List, Tuple, Union

import numpy as np

from robin.supply.entities import Service


class ConflictMatrix:
    """Symmetric boolean conflict matrix indexed by service ids.

    ``matrix[i, j]`` is ``True`` when the services with ids mapping to ``i``
    and ``j`` cannot be scheduled together (e.g. overlapping infrastructures
    within the safety headway).
    """

    def __init__(self, services: List[Service]) -> None:
        self.idx = {service.id: i for i, service in enumerate(services)}
        n = len(services)
        self.matrix: np.ndarray = np.zeros((n, n), dtype=bool)

    def set(self, row_id: str, col_id: str, value: bool) -> None:
        i = self.idx[row_id]
        j = self.idx[col_id]
        self.matrix[i, j] = value
        self.matrix[j, i] = value

    def get(self, row_id: str, col_id: str) -> bool:
        i = self.idx[row_id]
        j = self.idx[col_id]
        return bool(self.matrix[i, j])

    def toggle(self, row_id: str, col_id: str) -> None:
        """Flip the conflict flag between two services."""
        i, j = self.idx[row_id], self.idx[col_id]
        self.matrix[i, j] = not self.matrix[i, j]

    def row(self, row_id: str) -> np.ndarray:
        """Return the full boolean row for a service."""
        return self.matrix[self.idx[row_id], :]

    def col(self, col_id: str) -> np.ndarray:
        """Return the full boolean column for a service."""
        return self.matrix[:, self.idx[col_id]]


class Boundaries:
    """Lower/upper bounds for the real and discrete decision variables.

    Attributes:
        real: List with the ``(lower, upper)`` bounds for each real variable.
        discrete: List with the ``(lower, upper)`` bounds for each discrete
            variable.
    """

    def __init__(
        self,
        real: List[Union[Any, Tuple[float, float]]],
        discrete: List[Union[Any, Tuple[int, int]]],
    ) -> None:
        self.real = real
        self.discrete = discrete


class Solution:
    """A candidate solution split into its real and discrete parts.

    Attributes:
        real: Array with the real variables (e.g. departure times).
        discrete: Array with the discrete variables (e.g. scheduled mask).
    """

    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete
