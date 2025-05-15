"""Utils for all CRAFT sub-modules."""

import numpy as np

from robin.supply.entities import Service

from typing import Any, List, Tuple, Union


class ConflictMatrix:
    def __init__(self, services: List[Service]):
        self.idx = {service.id: i for i, service in enumerate(services)}
        n = len(services)
        self.matrix: np.ndarray = np.zeros((n, n), dtype=bool)

    def set(self, row_id: str, col_id: str, value: bool):
        i = self.idx[row_id]
        j = self.idx[col_id]
        self.matrix[i, j] = value
        self.matrix[j, i] = value

    def get(self, row_id: str, col_id: str) -> bool:
        i = self.idx[row_id]
        j = self.idx[col_id]
        return bool(self.matrix[i, j])

    def toggle(self, row_id: str, col_id: str):
        """Ejemplo de operación vectorizada mínima."""
        i, j = self.idx[row_id], self.idx[col_id]
        self.matrix[i, j] = not self.matrix[i, j]

    def row(self, row_id: str) -> np.ndarray:
        """Devuelve toda la fila booleana."""
        return self.matrix[self.idx[row_id], :]

    def col(self, col_id: str) -> np.ndarray:
        """Devuelve toda la columna booleana."""
        return self.matrix[:, self.idx[col_id]]


class Boundaries:
    """
    Boundaries

    This class contains the boundaries for the real and discrete variables.

    Attributes:
        real (List[Union[Any, Tuple[float, float]]]): List with the lower and upper bounds for each real variable
        discrete (List[Union[Any, Tuple[int, int]]]): List with the lower and upper bounds for each discrete variable
    """
    def __init__(
        self,
        real: List[Union[Any, Tuple[float, float]]],
        discrete: List[Union[Any, Tuple[int, int]]]
    ) -> None:
        """
        Initialize the Boundaries class

        Args:
            real (List[Union[Any, Tuple[float, float]]]): List with the lower and upper bounds for each real variable
            discrete (List[Union[Any, Tuple[int, int]]]): List with the lower and upper bounds for each discrete variable
        """
        self.real = real
        self.discrete = discrete


class Solution:
    """
    Solution

    This class contains the solution of the optimization algorithm.

    Attributes:
        real (np.ndarray): Array with the real variables
        discrete (np.ndarray): Array with the discrete variables
    """
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete
