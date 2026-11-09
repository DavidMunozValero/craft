"""Tests for the statistical comparison tools (``craft.stats``)."""

import numpy as np
import pytest

from craft.stats import cliffs_delta, effect_magnitude, holm_correction, wilcoxon_signed_rank


def test_cliffs_delta_complete_separation():
    """Fully ordered samples give the extreme values."""
    a = np.array([1.0, 2.0, 3.0])
    b = np.array([4.0, 5.0, 6.0])
    assert cliffs_delta(a, b) == pytest.approx(-1.0)
    assert cliffs_delta(b, a) == pytest.approx(1.0)


def test_cliffs_delta_identical_samples_is_zero():
    """Overlapping samples give a zero effect."""
    values = np.array([1.0, 2.0, 3.0, 4.0])
    assert cliffs_delta(values, values) == pytest.approx(0.0)


def test_cliffs_delta_known_case():
    """A one-of-four dominance difference gives the fractional value."""
    a = np.array([1.0, 5.0])
    b = np.array([2.0, 6.0])
    assert cliffs_delta(a, b) == pytest.approx(-0.5)


def test_effect_magnitude_bands():
    """The standard Romano-style bands, symmetric in the sign."""
    assert effect_magnitude(0.0) == "negligible"
    assert effect_magnitude(0.2) == "small"
    assert effect_magnitude(-0.4) == "medium"
    assert effect_magnitude(0.6) == "large"


def test_wilcoxon_detects_a_consistent_shift():
    """A constant positive shift over ten pairs is significant."""
    b = np.arange(10, dtype=float)
    a = b + 0.5
    statistic, p_value = wilcoxon_signed_rank(a, b)
    assert statistic == pytest.approx(0.0)
    assert p_value < 0.01


def test_wilcoxon_identical_samples_are_not_significant():
    """A null effect keeps the null hypothesis."""
    values = np.arange(8, dtype=float)
    _, p_value = wilcoxon_signed_rank(values, values)
    assert p_value == pytest.approx(1.0)


def test_wilcoxon_requires_paired_lengths():
    """Unpaired samples are rejected."""
    with pytest.raises(ValueError):
        wilcoxon_signed_rank([1.0, 2.0], [1.0, 2.0, 3.0])


def test_holm_correction_textbook_case():
    """The classic three-comparison example of the step-down procedure."""
    adjusted = holm_correction([0.01, 0.04, 0.03])
    assert adjusted == pytest.approx([0.03, 0.06, 0.06])


def test_holm_correction_caps_at_one_and_preserves_order():
    """Adjusted values never exceed one and keep the input order."""
    adjusted = holm_correction([0.9, 0.5, 0.2])
    assert adjusted == pytest.approx([1.0, 1.0, 0.6])


def test_holm_correction_single_comparison_is_identity():
    """One p-value maps to itself."""
    assert holm_correction([0.03]) == pytest.approx([0.03])
