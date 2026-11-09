"""Statistical comparison tools for the experiment battery.

Pairwise comparison of method samples evaluated on a shared seed set: the
Wilcoxon signed-rank test (paired by seed), the Cliff's delta effect size
with the standard magnitude bands, and the Holm step-down correction for
families of comparisons. scipy is imported lazily so that ``import craft``
stays light.

Conventions: samples are paired (method A and method B evaluated on the same
seeds, in the same order); the test is two-sided; Cliff's delta is positive
when A tends to exceed B.
"""

from typing import Sequence, Tuple

import numpy as np

MAGNITUDE_BANDS: Tuple[Tuple[float, str], ...] = (
    (0.147, "negligible"),
    (0.33, "small"),
    (0.474, "medium"),
)


def wilcoxon_signed_rank(a: Sequence[float], b: Sequence[float]) -> Tuple[float, float]:
    """Two-sided Wilcoxon signed-rank test on paired samples.

    Args:
        a: Sample of method A (one value per seed).
        b: Sample of method B, paired with ``a``.

    Returns:
        ``(statistic, p_value)``; ``(nan, 1.0)`` when every pair ties, which
        scipy's implementation rejects but is a decisive non-effect.

    Raises:
        ValueError: If the samples differ in length.
    """
    from scipy.stats import wilcoxon

    sample_a = np.asarray(a, dtype=float)
    sample_b = np.asarray(b, dtype=float)
    if sample_a.shape != sample_b.shape:
        raise ValueError("paired samples must have the same length")
    if np.allclose(sample_a - sample_b, 0.0):
        return float("nan"), 1.0
    statistic, p_value = wilcoxon(sample_a, sample_b, zero_method="zsplit")
    return float(statistic), float(p_value)


def cliffs_delta(a: Sequence[float], b: Sequence[float]) -> float:
    """Cliff's delta effect size of two samples (positive means A exceeds B).

    Args:
        a: Sample of method A.
        b: Sample of method B.

    Returns:
        Delta in ``[-1, 1]``: the probability of A exceeding B minus the
        probability of B exceeding A.
    """
    sample_a = np.asarray(a, dtype=float)
    sample_b = np.asarray(b, dtype=float)
    greater = sum(int((value > sample_b).sum()) for value in sample_a)
    less = sum(int((value < sample_b).sum()) for value in sample_a)
    return float((greater - less) / (sample_a.size * sample_b.size))


def effect_magnitude(delta: float) -> str:
    """Romano-style magnitude band of a Cliff's delta value.

    Args:
        delta: Effect size in ``[-1, 1]``.

    Returns:
        One of ``negligible``, ``small``, ``medium`` or ``large``.
    """
    for threshold, name in MAGNITUDE_BANDS:
        if abs(delta) < threshold:
            return name
    return "large"


def holm_correction(p_values: Sequence[float]) -> np.ndarray:
    """Holm step-down adjusted p-values, input order preserved.

    Args:
        p_values: Raw p-values of the comparison family.

    Returns:
        Adjusted p-values (monotone, capped at 1) in the same order as the
        input.
    """
    p = np.asarray(p_values, dtype=float)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p, kind="stable")
    adjusted_sorted = np.minimum(
        np.maximum.accumulate((m - np.arange(m)) * p[order]), 1.0
    )
    adjusted = np.empty(m)
    adjusted[order] = adjusted_sorted
    return adjusted
