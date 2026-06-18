"""Miscellaneous analysis helpers for CRAFT experiments.

Post-optimization utilities to aggregate revenue by RU and to compare
candidate solutions, used by the analysis notebooks.
"""

from typing import Mapping

import numpy as np
import pandas as pd

from robin.supply.entities import Supply


def get_rus_revenue(supply: Supply, df: pd.DataFrame) -> Mapping[str, float]:
    """Aggregate revenue by RU from a robin output DataFrame.

    Args:
        supply: Supply object providing the service -> TSP mapping.
        df: DataFrame from robin's output data with columns 'service' and
            'price'.

    Returns:
        Mapping from RU (TSP) name to its total revenue.
    """
    services_tsp = {service.id: service.tsp.name for service in supply.services}
    df['tsp'] = df['service'].apply(lambda service_id: services_tsp.get(service_id, np.nan))
    tsp_revenue = df.groupby('tsp').agg({'price': 'sum'}).to_dict()['price']
    return tsp_revenue


def is_better_solution(
    rus_revenue: Mapping[str, float],
    best_solution: Mapping[str, float]
) -> bool:
    """Check if the current solution is better than the best solution so far.

    A solution is better if there is no best yet, it schedules more RUs, or
    it improves the revenue of at least half of the RUs.

    Args:
        rus_revenue: Revenue of each RU for the current solution.
        best_solution: Best solution found so far.

    Returns:
        True if the current solution is better, False otherwise.
    """
    if not best_solution:
        return True
    elif len(rus_revenue) > len(best_solution):
        return True
    elif sum([rus_revenue[tsp] > best_solution.get(tsp, -np.inf) for tsp in rus_revenue]) >= len(rus_revenue) // 2:
        return True
    return False
