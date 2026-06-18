"""Tests for craft.common: Boundaries, Solution, ConflictMatrix."""

import numpy as np
import pytest

from craft.common import Boundaries, ConflictMatrix, Solution


class TestBoundaries:
    def test_real_and_discrete_stored(self):
        b = Boundaries(real=[(0.0, 10.0), (5.0, 15.0)], discrete=[(0, 1), (0, 5)])
        assert b.real == [(0.0, 10.0), (5.0, 15.0)]
        assert b.discrete == [(0, 1), (0, 5)]

    def test_empty_boundaries(self):
        b = Boundaries(real=[], discrete=[])
        assert b.real == []
        assert b.discrete == []


class TestSolution:
    def test_real_and_discrete_stored(self):
        s = Solution(real=np.array([1.0, 2.0]), discrete=np.array([1, 0, 1]))
        assert np.array_equal(s.real, [1.0, 2.0])
        assert np.array_equal(s.discrete, [1, 0, 1])

    def test_empty_solution(self):
        s = Solution(real=np.array([]), discrete=np.array([]))
        assert s.real.size == 0
        assert s.discrete.size == 0


class TestConflictMatrix:
    @pytest.fixture
    def services(self):
        class FakeService:
            def __init__(self, sid):
                self.id = sid
        return [FakeService("a"), FakeService("b"), FakeService("c")]

    def test_initial_matrix_all_false(self, services):
        cm = ConflictMatrix(services=services)
        assert cm.matrix.shape == (3, 3)
        assert not cm.matrix.any()

    def test_set_symmetric(self, services):
        cm = ConflictMatrix(services=services)
        cm.set("a", "b", True)
        assert cm.get("a", "b") is True
        assert cm.get("b", "a") is True
        assert cm.get("a", "c") is False

    def test_toggle(self, services):
        cm = ConflictMatrix(services=services)
        cm.set("a", "b", True)
        cm.toggle("a", "b")
        assert cm.get("a", "b") is False

    def test_row(self, services):
        cm = ConflictMatrix(services=services)
        cm.set("a", "b", True)
        cm.set("a", "c", True)
        row = cm.row("a")
        assert np.array_equal(row, [False, True, True])

    def test_col(self, services):
        cm = ConflictMatrix(services=services)
        cm.set("a", "b", True)
        col = cm.col("b")
        assert np.array_equal(col, [True, False, False])
