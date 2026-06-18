"""Tests for craft.gsa: fields, elements, and the GSA optimizer."""

import numpy as np
import pytest

from craft.common import Boundaries, Solution
from craft.gsa import GSA
from craft.gsa.elements import Acceleration, GConstant, Velocity
from craft.gsa.fields import (
    g_bin_constant,
    g_field,
    g_real_constant,
    mass_calculation,
    sin_chaotic_term,
)


class TestMassCalculation:
    def test_uniform_fitness(self):
        fit = np.array([5.0, 5.0, 5.0])
        mass = mass_calculation(fit)
        assert np.allclose(mass, 1 / 3)

    def test_non_uniform_fitness(self):
        fit = np.array([0.0, 1.0])
        mass = mass_calculation(fit)
        assert mass.sum() == pytest.approx(1.0)
        assert mass[1] > mass[0]

    def test_single_agent(self):
        fit = np.array([42.0])
        mass = mass_calculation(fit)
        assert np.allclose(mass, 1.0)


class TestGravitationalConstants:
    def test_g_bin_constant_decreasing(self):
        g0 = g_bin_constant(0, 10)
        g5 = g_bin_constant(5, 10)
        g10 = g_bin_constant(10, 10)
        assert g0 == 1.0
        assert g5 == pytest.approx(0.5)
        assert g10 == pytest.approx(0.0)

    def test_g_real_constant_decreasing(self):
        g0 = g_real_constant(0, 10)
        g5 = g_real_constant(5, 10)
        assert g0 == 100.0
        assert g5 < g0
        assert g5 > 0


class TestSinChaoticTerm:
    def test_returns_tuple(self):
        result = sin_chaotic_term(5, 10.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_zero_iteration(self):
        val, x = sin_chaotic_term(0, 10.0)
        assert x == pytest.approx(0.7)


class TestGField:
    def test_empty_dim(self):
        acc = g_field(
            population_size=5,
            dim=0,
            pos=np.array([]),
            mass=np.ones(5),
            current_iter=0,
            max_iters=10,
            gravity_constant=1.0,
            r_power=1,
            elitist_check=True,
            real=True,
        )
        assert acc.size == 0

    def test_real_field_shape(self):
        pop, dim = 5, 3
        pos = np.random.rand(pop, dim) * 100
        mass = np.ones(pop) / pop
        acc = g_field(
            population_size=pop,
            dim=dim,
            pos=pos,
            mass=mass,
            current_iter=0,
            max_iters=10,
            gravity_constant=100.0,
            r_power=1,
            elitist_check=True,
            real=True,
        )
        assert acc.shape == (pop, dim)

    def test_discrete_field_shape(self):
        pop, dim = 5, 4
        pos = np.random.randint(0, 2, size=(pop, dim)).astype(float)
        mass = np.ones(pop) / pop
        acc = g_field(
            population_size=pop,
            dim=dim,
            pos=pos,
            mass=mass,
            current_iter=0,
            max_iters=10,
            gravity_constant=1.0,
            r_power=1,
            elitist_check=True,
            real=False,
        )
        assert acc.shape == (pop, dim)


class TestElements:
    def test_velocity(self):
        v = Velocity(real=np.array([1.0, 2.0]), discrete=np.array([0, 1]))
        assert np.array_equal(v.real, [1.0, 2.0])
        assert np.array_equal(v.discrete, [0, 1])

    def test_acceleration(self):
        a = Acceleration(real=np.array([0.5]), discrete=np.array([0.3]))
        assert a.real == pytest.approx(0.5)

    def test_gconstant(self):
        g = GConstant(real=100.0, discrete=1.0)
        assert g.real == 100.0
        assert g.discrete == 1.0


class TestGSA:
    def test_construct(self):
        def obj(sol):
            return (0.0, 0.0)

        bounds = Boundaries(real=[(0.0, 1.0)] * 3, discrete=[])
        gsa = GSA(objective_function=obj, r_dim=3, d_dim=0, boundaries=bounds)
        assert gsa.r_dim == 3
        assert gsa.d_dim == 0
        assert gsa.t_dim == 3

    def test_optimize_convergence(self):
        def obj_sphere(sol):
            val = float(-np.sum((sol.real - 50.0) ** 2))
            return (val, 0.0)

        bounds = Boundaries(real=[(0.0, 100.0)] * 3, discrete=[])
        gsa = GSA(objective_function=obj_sphere, r_dim=3, d_dim=0, boundaries=bounds)
        history = gsa.optimize(population_size=10, iters=15, seed=42, verbose=False)

        assert len(history) == 15
        assert gsa.best_fitness > float("-inf")
        assert len(gsa.convergence) == 15
        assert len(gsa.solution_history) == 15
        assert isinstance(gsa.best_solution, Solution)
        assert gsa.best_solution.real.shape == (3,)

    def test_optimize_reproducible(self):
        def obj(sol):
            return (float(np.sum(sol.real)), 0.0)

        bounds = Boundaries(real=[(0.0, 10.0)] * 4, discrete=[])
        g1 = GSA(objective_function=obj, r_dim=4, d_dim=0, boundaries=bounds)
        g1.optimize(population_size=8, iters=10, seed=99, verbose=False)
        g2 = GSA(objective_function=obj, r_dim=4, d_dim=0, boundaries=bounds)
        g2.optimize(population_size=8, iters=10, seed=99, verbose=False)
        assert g1.best_fitness == g2.best_fitness
        assert np.allclose(g1.best_solution.real, g2.best_solution.real)

    def test_optimize_discrete(self):
        def obj(sol):
            return (float(np.sum(sol.discrete)), 0.0)

        bounds = Boundaries(real=[], discrete=[(0, 1)] * 6)
        gsa = GSA(objective_function=obj, r_dim=0, d_dim=6, boundaries=bounds)
        gsa.optimize(population_size=10, iters=10, seed=42, verbose=False)
        assert gsa.best_fitness > 0
        assert str(gsa.best_solution.discrete.dtype) in ("int64", "int32", "int")

    def test_convergence_non_decreasing(self):
        def obj(sol):
            return (float(-np.sum((sol.real - 50.0) ** 2)), 0.0)

        bounds = Boundaries(real=[(0.0, 100.0)] * 3, discrete=[])
        gsa = GSA(objective_function=obj, r_dim=3, d_dim=0, boundaries=bounds)
        gsa.optimize(population_size=10, iters=10, seed=42, verbose=False)
        for i in range(len(gsa.convergence) - 1):
            assert gsa.convergence[i] <= gsa.convergence[i + 1] + 1e-9
