"""Utils for all CRAFT sub-modules."""

import numpy as np

from robin.supply.entities import Service, Supply

from geopy.distance import geodesic
from scipy.stats import loguniform
from typing import Any, List, Mapping, Tuple, Union


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


class RevenueSimulator:
    """
    RevenueSimulator

    This class contains the revenue simulator for the optimization algorithm.

    Attributes:
        real (np.ndarray): Array with the real variables
        discrete (np.ndarray): Array with the discrete variables
    """
    def __init__(self, supply: Supply) -> None:
        self.supply = supply

    def simulate_revenue(self, alpha: float = 2 / 3) -> Mapping[str, Mapping[str, float]]:
        """
        Calculate revenue behavior parameters for each service in the supply.

        For every service in the supply, this function computes:
          - canon: The base revenue increased by a randomly selected bias factor.
          - k: A random scaling factor drawn from a log-uniform distribution.
          - dt_max_penalty: A penalty value derived from the canon.
          - tt_max_penalty: A penalty value distributed across the service's stations.
          - importance: A normalized random weight assigned to the service within its RU group.

        The services are first grouped by their associated RU (Transport Service Provider).
        Then, for each group, a set of random values is generated and normalized so that the
        sum of importance values in each group is 1. Finally, these importance values are assigned
        to the corresponding services.

        Args:
            supply (Supply): An object containing a list of services. Each service is expected to have:
                - service.id: A unique identifier for the service.
                - service.line.stations: A collection (e.g., list) of stations.
                - service.tsp.id: The identifier of the associated RU.

        Returns:
            Mapping[str, Dict[str, float]]:
                A dictionary mapping each service's ID to a dictionary of computed revenue parameters.
                Each dictionary contains:
                    - 'canon': Computed base revenue (float).
                    - 'ru': The RU identifier (same as service.tsp.id).
                    - 'k': A random scaling factor (float).
                    - 'dt_max_penalty': Penalty value for DT (float).
                    - 'tt_max_penalty': Penalty value for TT (float).
                    - 'importance': Normalized importance weight (float) within its RU group.
        """
        revenue = {}
        tsp_k = {}
        for service in self.supply.services:
            sta_coords = [sta.coordinates for sta in service.line.stations]
            distances = [geodesic(sta_coords[i], sta_coords[i + 1]).km for i in range(len(sta_coords) - 1)]
            distance_factor = 7 * sum(distances)
            service_capacity = service.rolling_stock.total_capacity
            capacity_factor = (alpha * service_capacity) / 100 * 1.67
            stations_factor = 18 + (len(sta_coords) - 2) * 65 + 165
            total_canon = (distance_factor + capacity_factor + stations_factor) / 100
            max_penalty = total_canon * 0.3
            dt_penalty = np.round(max_penalty * 0.35, 2)
            tt_penalty = np.round((max_penalty - dt_penalty) / (len(sta_coords) - 1), 2)
            if service.tsp.id not in tsp_k:
                k = np.round(loguniform.rvs(0.01, 100, 1), 2)
                tsp_k[service.tsp.id] = k
            else:
                k = tsp_k[service.tsp.id]

            revenue[service.id] = {
                'canon': total_canon,
                'ru': service.tsp.id,
                'k': k,
                'dt_max_penalty': dt_penalty,
                'tt_max_penalty': tt_penalty
            }

        return revenue


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
