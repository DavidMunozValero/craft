"""Quality indicators for bi-objective Pareto fronts (maximization).

Measurement tools for comparing the Pareto fronts produced by different
methods (own NSGA-II, external baselines, scalar sweeps, exact reference
points):

* :func:`normalized_hypervolume` — hypervolume of the front after min-max
  normalization against a reference set, with the reference point at the
  nadir corner; scale-free and comparable across instances, seeds and
  methods.
* :func:`igd` — inverted generational distance against a reference set
  (lower is better; zero when the front covers the reference exactly).
* :func:`spread` — Deb's delta, the relative spread of the front with
  respect to the extreme points of a reference set (lower is better).
* :func:`c_metric` — set coverage of one front by another (higher is
  better; 1.0 when every point of the second front is weakly dominated).

Every indicator assumes maximization of both objectives, the convention of
the timetabling formulation (revenue and fairness). The reference set is
typically the union of the fronts under comparison, enriched with exact
epsilon-constraint points when available. Because :func:`igd` and
:func:`spread` build on Euclidean distances, objectives on very different
scales (revenue vs fairness) should be brought to a common range with
:func:`min_max_normalize` before calling them.
"""

import numpy as np

from .nsga2 import hypervolume


def ideal(values: np.ndarray) -> np.ndarray:
    """Component-wise maximum of a front (best corner under maximization).

    Args:
        values: Objective matrix, shape ``(n, m)``.

    Returns:
        Vector of length ``m`` with the best value attained per objective.
    """
    values = np.asarray(values, dtype=float)
    return np.asarray(values.max(axis=0), dtype=float)


def nadir(values: np.ndarray) -> np.ndarray:
    """Component-wise minimum of a front (worst corner under maximization).

    Args:
        values: Objective matrix, shape ``(n, m)``.

    Returns:
        Vector of length ``m`` with the worst value attained per objective.
    """
    values = np.asarray(values, dtype=float)
    return np.asarray(values.min(axis=0), dtype=float)


def min_max_normalize(
    values: np.ndarray, low: np.ndarray, high: np.ndarray
) -> np.ndarray:
    """Scale objective vectors from ``[low, high]`` to ``[0, 1]`` per objective.

    Degenerate spans (``high == low``) are treated as 1 to avoid division
    by zero. Values outside the bounds keep their relative position
    (below ``low`` becomes negative, above ``high`` exceeds 1).

    Args:
        values: Objective matrix, shape ``(n, m)``.
        low: Lower bound per objective, shape ``(m,)``.
        high: Upper bound per objective, shape ``(m,)``.

    Returns:
        The normalized matrix, same shape as ``values``.
    """
    values = np.asarray(values, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    span = np.where(high - low > 0, high - low, 1.0)
    return np.asarray((values - low) / span, dtype=float)


def normalized_hypervolume(front: np.ndarray, reference: np.ndarray) -> float:
    """Hypervolume of the front normalized against a reference set.

    The front is scaled per objective with the nadir and ideal corners of
    ``reference`` and its hypervolume is computed with the reference point
    at the origin of the normalized space (the nadir corner). A front that
    spans the whole reference extent scores 1.0; points falling below the
    nadir contribute nothing, and points beyond the ideal may push the
    value above 1.0.

    Args:
        front: Objective matrix of the front, shape ``(n, 2)``.
        reference: Objective matrix of the reference set (e.g. the union of
            all compared fronts), shape ``(k, 2)``.

    Returns:
        The hypervolume in normalized units.
    """
    front = np.asarray(front, dtype=float)
    reference = np.asarray(reference, dtype=float)
    normalized = min_max_normalize(front, nadir(reference), ideal(reference))
    return hypervolume(normalized, (0.0, 0.0))


def igd(front: np.ndarray, reference: np.ndarray) -> float:
    """Inverted generational distance of a front against a reference set.

    Mean Euclidean distance from each reference point to its nearest front
    point. Zero when the front covers the reference exactly; smaller is
    better. Normalize the inputs with :func:`min_max_normalize` when the
    objectives live on different scales.

    Args:
        front: Objective matrix of the front, shape ``(n, m)``.
        reference: Reference objective matrix, shape ``(k, m)``.

    Returns:
        The mean nearest-neighbour distance from ``reference`` to ``front``.

    Raises:
        ValueError: If the front or the reference set is empty.
    """
    front = np.asarray(front, dtype=float)
    reference = np.asarray(reference, dtype=float)
    if front.size == 0:
        raise ValueError("the front must not be empty")
    if reference.size == 0:
        raise ValueError("the reference set must not be empty")
    distances = np.linalg.norm(reference[:, None, :] - front[None, :, :], axis=2)
    return float(distances.min(axis=1).mean())


def spread(front: np.ndarray, reference: np.ndarray | None = None) -> float:
    """Deb's spread delta of a front.

    Measures how uniformly the front covers its extent. Consecutive
    Euclidean distances between the front points (sorted by the first
    objective) are compared with their mean, and the gap to the extreme
    points of ``reference`` (when given) is added to both terms:

    ``Delta = (d_f + d_l + sum |d_i - mean(d)|) / (d_f + d_l + sum d_i)``

    Zero means perfect uniformity and full extent; one means the degenerate
    case (a lone point or every point clustered together). With no
    reference the extremes terms vanish and only internal uniformity is
    measured. Normalize the inputs with :func:`min_max_normalize` when the
    objectives live on different scales.

    Args:
        front: Objective matrix of the front, shape ``(n, m)``.
        reference: Optional objective matrix providing the extreme points
            (per-objective best) the front should reach.

    Returns:
        The spread delta in ``[0, 1]``.

    Raises:
        ValueError: If the front is empty.
    """
    front = np.asarray(front, dtype=float)
    if front.size == 0:
        raise ValueError("the front must not be empty")
    ordered = front[np.argsort(front[:, 0], kind="stable")]
    if len(ordered) < 2:
        return 1.0
    consecutive = np.linalg.norm(np.diff(ordered, axis=0), axis=1)
    mean = consecutive.mean()
    extremes = 0.0
    if reference is not None:
        reference = np.asarray(reference, dtype=float)
        f1_extreme = reference[int(np.argmax(reference[:, 0]))]
        f2_extreme = reference[int(np.argmax(reference[:, 1]))]
        extremes = float(
            np.linalg.norm(f2_extreme - ordered[0])
            + np.linalg.norm(f1_extreme - ordered[-1])
        )
    denominator = extremes + consecutive.sum()
    if denominator == 0.0:
        return 1.0
    deviation = np.abs(consecutive - mean).sum()
    return float((extremes + deviation) / denominator)


def c_metric(a: np.ndarray, b: np.ndarray) -> float:
    """Coverage of front ``b`` by front ``a`` (Zitzler's C-metric).

    Fraction of the points of ``b`` that are weakly dominated by at least
    one point of ``a`` (equal points count as covered). One means ``a``
    covers ``b`` entirely, zero means no point of ``b`` is covered; two
    mutually non-dominated fronts score zero on both directions.

    Args:
        a: Objective matrix of the covering front, shape ``(n, m)``.
        b: Objective matrix of the covered front, shape ``(k, m)``.

    Returns:
        The covered fraction of ``b`` in ``[0, 1]``.

    Raises:
        ValueError: If ``b`` is empty.
    """
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if b.size == 0:
        raise ValueError("the covered front must not be empty")
    covered = (a[:, None, :] >= b[None, :, :]).all(axis=2).any(axis=0)
    return float(covered.mean())
