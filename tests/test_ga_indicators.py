"""Tests for the Pareto-front quality indicators (``craft.ga.indicators``)."""

import numpy as np
import pytest
from numpy.testing import assert_allclose

from craft.ga.indicators import (
    c_metric,
    ideal,
    igd,
    nadir,
    min_max_normalize,
    normalized_hypervolume,
    spread,
)
from craft.ga.nsga2 import hypervolume

DIAGONAL = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])


def test_ideal_and_nadir_corners():
    """Ideal keeps the best per objective and nadir the worst."""
    values = np.array([[1.0, 5.0], [3.0, 3.0], [5.0, 1.0]])
    assert_allclose(ideal(values), [5.0, 5.0])
    assert_allclose(nadir(values), [1.0, 1.0])


def test_min_max_normalize_maps_bounds_to_unit_range():
    """The bounds map to 0 and 1; other points scale linearly."""
    values = np.array([[2.0, 0.0], [0.0, 2.0], [1.0, 1.0]])
    normalized = min_max_normalize(values, low=[0.0, 0.0], high=[2.0, 2.0])
    assert_allclose(normalized, [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])


def test_min_max_normalize_keeps_relative_position_outside_bounds():
    """Values beyond the bounds are not clipped."""
    normalized = min_max_normalize(
        np.array([[3.0, -1.0]]), low=[0.0, 0.0], high=[2.0, 2.0]
    )
    assert_allclose(normalized, [[1.5, -0.5]])


def test_min_max_normalize_survives_degenerate_span():
    """A constant objective divides by one instead of by zero."""
    normalized = min_max_normalize(
        np.array([[1.0, 1.0]]), low=[1.0, 1.0], high=[1.0, 1.0]
    )
    assert_allclose(normalized, [[0.0, 0.0]])


def test_normalized_hypervolume_matches_raw_hypervolume_on_unit_reference():
    """With a reference spanning the unit square the normalization is the identity."""
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert normalized_hypervolume(DIAGONAL, reference) == pytest.approx(
        hypervolume(DIAGONAL, (0.0, 0.0))
    )


def test_normalized_hypervolume_known_area():
    """A staircase over the full reference extent scores three quarters."""
    reference = np.array([[2.0, 0.0], [0.0, 2.0]])
    front = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert normalized_hypervolume(front, reference) == pytest.approx(0.75)


def test_normalized_hypervolume_ignores_points_below_the_nadir():
    """Reference points falling under the nadir corner add nothing."""
    reference = np.array([[2.0, 0.0], [0.0, 2.0]])
    with_extra = np.array([[2.0, 1.0], [1.0, 2.0], [-1.0, 5.0]])
    without_extra = np.array([[2.0, 1.0], [1.0, 2.0]])
    assert normalized_hypervolume(with_extra, reference) == pytest.approx(
        normalized_hypervolume(without_extra, reference)
    )


def test_normalized_hypervolume_empty_front_is_zero():
    """An empty front covers no area."""
    assert normalized_hypervolume(np.empty((0, 2)), DIAGONAL) == 0.0


def test_igd_is_zero_when_front_covers_reference():
    """A front identical to the reference is at distance zero."""
    assert igd(DIAGONAL, DIAGONAL) == pytest.approx(0.0)


def test_igd_known_case():
    """Only the uncovered middle reference point contributes distance."""
    front = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert igd(front, DIAGONAL) == pytest.approx(np.sqrt(0.5) / 3.0)


def test_igd_prefers_the_closer_front():
    """The indicator decreases as the front approaches the reference."""
    reference = np.array([[0.5, 0.5]])
    near = np.array([[0.6, 0.6]])
    far = np.array([[1.0, 1.0]])
    assert igd(near, reference) < igd(far, reference)


def test_igd_rejects_empty_fronts():
    """Both operands are required."""
    with pytest.raises(ValueError):
        igd(np.empty((0, 2)), DIAGONAL)
    with pytest.raises(ValueError):
        igd(DIAGONAL, np.empty((0, 2)))


def test_spread_uniform_front_is_zero_without_reference():
    """Equally spaced points on the front give zero deviation."""
    assert spread(DIAGONAL) == pytest.approx(0.0)


def test_spread_uneven_front_is_positive():
    """A front with clustered points shows a positive deviation term."""
    front = np.array([[0.0, 1.0], [0.9, 0.2], [1.0, 0.0]])
    assert spread(front) == pytest.approx(0.6868, abs=1e-3)


def test_spread_single_point_with_reference_is_one():
    """A lone point is the degenerate worst spread."""
    reference = np.array([[1.0, 0.0], [0.0, 1.0]])
    assert spread(np.array([[0.5, 0.5]]), reference) == pytest.approx(1.0)


def test_spread_with_reference_penalizes_missing_extremes():
    """Reaching the reference extremes lowers the delta."""
    reference = np.array([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]])
    tight = np.array([[0.5, 0.5]])
    wide = DIAGONAL
    assert spread(tight, reference) > spread(wide, reference)
    assert spread(wide, reference) == pytest.approx(0.0)


def test_c_metric_full_and_zero_coverage():
    """Dominating fronts cover completely and are not covered back."""
    a = np.array([[1.0, 1.0]])
    b = np.array([[0.5, 0.5]])
    assert c_metric(a, b) == pytest.approx(1.0)
    assert c_metric(b, a) == pytest.approx(0.0)


def test_c_metric_partial_coverage():
    """Only the dominated points of the covered front count."""
    a = np.array([[1.0, 1.0]])
    b = np.array([[0.5, 0.5], [2.0, 0.0]])
    assert c_metric(a, b) == pytest.approx(0.5)


def test_c_metric_counts_weak_domination():
    """Identical points are covered, so shared points score one."""
    assert c_metric(DIAGONAL, DIAGONAL) == pytest.approx(1.0)


def test_c_metric_rejects_empty_covered_front():
    """The covered front is required."""
    with pytest.raises(ValueError):
        c_metric(DIAGONAL, np.empty((0, 2)))
