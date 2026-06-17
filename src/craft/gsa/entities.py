"""GSA - Gravitational Search Algorithm implementation."""

import math
import numpy as np
import pandas as pd
import random
import time
import os

from functools import lru_cache
from scipy.spatial.distance import euclidean, hamming
from typing import Any, List, Mapping, Tuple, Union

from copy import deepcopy


class Boundaries:
    def __init__(
        self,
        real: List[Union[Any, Tuple[float, float]]],
        discrete: List[Union[Any, Tuple[int, int]]]
    ) -> None:
        self.real = real
        self.discrete = discrete


class Solution:
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class Velocity:
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class Acceleration:
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


class GConstant:
    def __init__(self, real, discrete) -> None:
        self.real = real
        self.discrete = discrete


def mass_calculation(fit: np.ndarray) -> np.ndarray:
    f_min, f_max = fit.min(), fit.max()
    if f_max == f_min:
        return np.ones(fit.shape) / len(fit)
    normalized_fit = (fit - f_min) / (f_max - f_min)
    mass = normalized_fit / normalized_fit.sum()
    return mass


def g_bin_constant(curr_iter: int, max_iters: int, g_zero: float = 1) -> float:
    return g_zero * (1 - (curr_iter / max_iters))


def g_real_constant(curr_iter: int, max_iters: int, alpha: float = 20, g_zero: float = 100) -> float:
    return g_zero * np.exp(- ((alpha * curr_iter) / max_iters))


@lru_cache(maxsize=None)
def compute_x(i: int) -> float:
    if i == 0:
        return 0.7
    prev_x = compute_x(i - 1)
    return 2.3 * prev_x ** 2 * np.sin(np.pi * prev_x)


def sin_chaotic_term(curr_iter: int, value: float) -> Tuple[float, float]:
    x = compute_x(curr_iter)
    return x * value, x


def g_field(
    population_size: int,
    dim: int,
    pos: np.ndarray,
    mass: np.ndarray,
    current_iter: int,
    max_iters: int,
    gravity_constant: float,
    r_power: int,
    elitist_check: bool,
    real: bool
) -> np.ndarray:
    """Calculate gravitational field."""
    if dim == 0:
        return np.array([])
    
    pos = np.asarray(pos, dtype=float)
    if pos.size == 0:
        return np.array([])
    
    if pos.ndim == 1:
        if pos.shape[0] == population_size * dim:
            pos = pos.reshape(population_size, dim)
        elif pos.shape[0] == dim:
            pos = pos.reshape(1, dim)
        else:
            raise ValueError(f"Cannot reshape pos of shape {pos.shape} to ({population_size}, {dim})")
    if not dim > 0:
        return np.array([])

    final_per = 2
    if elitist_check:
        k_best = final_per + (1 - current_iter / max_iters) * (100 - final_per)
        k_best = round(population_size * k_best / 100)
    else:
        k_best = population_size

    ds = sorted(range(len(mass)), key=lambda k: mass[k], reverse=True)

    acc = np.zeros((population_size, dim))

    for r in range(population_size):
        for ii in range(k_best):
            z = ds[ii]
            if z != r:
                x = pos[r, :]
                y = pos[z, :]
                if real:
                    radius = euclidean(x, y)
                else:
                    radius = hamming(x, y)

                for k in range(dim):
                    n = random.random()
                    acc[r, k] += n * gravity_constant * (mass[z] / (radius + np.finfo(float).eps)) * (pos[z, k] - pos[r, k])

    return acc


class GSA:
    def __init__(
        self,
        objective_function: callable,
        r_dim: int,
        d_dim: int,
        boundaries: Boundaries,
        is_feasible: Union[callable, None] = None,
        custom_repair: Union[None, callable] = None
    ) -> None:
        self.objective_function = objective_function
        if is_feasible is None:
            self.is_feasible = lambda _: True
        self.is_feasible = is_feasible
        self.custom_repair = custom_repair
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.t_dim = self.r_dim + self.d_dim
        self.boundaries = boundaries

        self.population_history = pd.DataFrame()
        self.objective_function_name = self.objective_function.__name__
        self.solution_history = None
        self.accuracy_history = None
        self.convergence = None
        self.start_time = None
        self.end_time = None
        self.execution_time = None

    def _get_initial_positions(self, population_size: int) -> List[Solution]:
        pos_r = np.array([
            np.random.uniform(low=rd_lb, high=rd_ub, size=population_size)
            for rd_lb, rd_ub in self.boundaries.real
        ]).T

        pos_d = np.array([
            np.random.choice(a=range(dd_lb, dd_ub + 1), size=population_size)
            for dd_lb, dd_ub in self.boundaries.discrete
        ]).T

        population = []
        for sol in range(population_size):
            real_part = pos_r[sol, :] if self.r_dim > 0 else np.array([])
            discrete_part = pos_d[sol, :] if self.d_dim > 0 else np.array([])
            
            iters = 0
            while not self.is_feasible(Solution(real=real_part, discrete=discrete_part)) and iters < 100:
                if self.r_dim > 0:
                    real_part = np.array([
                        np.random.uniform(low=rd_lb, high=rd_ub)
                        for rd_lb, rd_ub in self.boundaries.real
                    ])
                if self.d_dim > 0:
                    discrete_part = np.array([
                        np.random.choice(a=range(dd_lb, dd_ub + 1))
                        for dd_lb, dd_ub in self.boundaries.discrete
                    ])
                iters += 1
            
            solution = Solution(real=real_part, discrete=discrete_part)
            population.append(solution)
        return population

    def optimize(
        self,
        population_size: int,
        iters: int,
        r_power: int = 1,
        elitist_check: bool = True,
        chaotic_constant: bool = False,
        repair_solution: bool = False,
        initial_population: Union[None, List[Solution]] = None,
        w_max: float = 20.0,
        w_min: float = 1e-10,
        save_population: bool = False,
        verbose: bool = True
    ) -> pd.DataFrame:
        vel = [
            Solution(np.zeros(self.r_dim, dtype=np.float64), np.zeros(self.d_dim, dtype=np.float64))
            for _ in range(population_size)
        ]
        fit = np.zeros(population_size, dtype=np.float64)
        mass = np.zeros(population_size, dtype=np.float64)

        g_best = Solution(np.zeros(self.r_dim, dtype=np.int64), np.zeros(self.d_dim, dtype=np.int64))
        g_best_score = float("-inf")
        best_acc = 0.0

        if initial_population is None:
            pos = self._get_initial_positions(population_size)
        else:
            pos = initial_population

        best_solution_history = []
        convergence_curve = np.zeros(iters)

        timer_start = time.time()
        self.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        columns = ['Iteration', 'Fitness', 'Accuracy', 'ExecutionTime', 'Discrete', 'Real']
        history = pd.DataFrame(columns=columns)

        print(f"GSA starting: {population_size} agents, {iters} iterations")
        
        for current_iter in range(iters):
            print(f"First iteration starting...")
            
            for i in range(population_size):
                solution = pos[i]
                # print(f"  Agent {i}: real type={type(solution.real)}, shape={getattr(solution.real, 'shape', 'N/A')}")
                fitness, accuracy = self.objective_function(solution)
                # print(f"  Agent {i} done, fitness={fitness}")
                fit[i] = fitness
            
            print(f"All {population_size} agents evaluated, best={max(fit)}")

            # Debug print
            print(f"Computing mass...")
            mass = mass_calculation(fit=fit)
            print(f"Mass computed: {mass[:3]}...")

            # Debug print
            print(f"Computing gravity constant...")

            gravity_constant = self._calculate_gravitational_constants(
                current_iter=current_iter,
                max_iters=iters,
                chaotic_constant=chaotic_constant,
                w_max=w_max,
                w_min=w_min
            )

            acc = self._calculate_acceleration(
                population_size=population_size,
                pos=pos,
                mass=mass,
                current_iter=current_iter,
                max_iters=iters,
                gravity_constant=gravity_constant,
                r_power=r_power,
                elitist_check=elitist_check
            )
            print(f"Acceleration computed")
            
            # Debug print
            print("Moving agents...")
            pos, vel = self._move(
                position=pos,
                velocity=vel,
                acceleration=acc,
                population=population_size,
                v_max=6,
                repair_solution=repair_solution
            )

            convergence_curve[current_iter] = g_best_score
            best_solution_history.append(g_best)

            if verbose:
                print(f'At iteration {current_iter + 1} the best fitness is {g_best_score}')

            for i, individual in enumerate(pos):
                if not self.is_feasible(individual):
                    print(f"Individual {i} is not feasible")

        timer_end = time.time()
        self.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.execution_time = timer_end - timer_start
        self.convergence = convergence_curve
        self.solution_history = best_solution_history

        if verbose:
            print(g_best)

        return history

    @staticmethod
    def _calculate_gravitational_constants(
        current_iter: int,
        max_iters: int,
        chaotic_constant: bool,
        w_max: float,
        w_min: float
    ) -> GConstant:
        g_real = g_real_constant(current_iter, max_iters)
        g_discrete = g_bin_constant(current_iter, max_iters)

        if chaotic_constant:
            ch_value = w_max - current_iter * ((w_max - w_min) / max_iters)
            chaotic_term, _ = sin_chaotic_term(current_iter, ch_value)
            g_real += chaotic_term
            g_discrete += chaotic_term

        return GConstant(real=g_real, discrete=g_discrete)

    def _calculate_acceleration(
        self,
        population_size: int,
        pos: List[Solution],
        mass: np.ndarray,
        current_iter: int,
        max_iters: int,
        gravity_constant: GConstant,
        r_power: int,
        elitist_check: bool = True
    ) -> List[Acceleration]:
        real_arr = np.array([p.real for p in pos], dtype=float)
        
        if real_arr.ndim == 1:
            if self.r_dim > 0 and real_arr.shape[0] == population_size * self.r_dim:
                real_arr = real_arr.reshape(population_size, self.r_dim)
            elif real_arr.shape[0] == self.r_dim:
                real_arr = real_arr.reshape(1, self.r_dim)
        
        if self.r_dim > 0:
            acc_r = g_field(
                population_size=population_size,
                dim=self.r_dim,
                pos=real_arr,
                mass=mass,
                current_iter=current_iter,
                max_iters=max_iters,
                gravity_constant=gravity_constant.real,
                r_power=r_power,
                elitist_check=elitist_check,
                real=True
            )
        else:
            acc_r = []

        if self.d_dim > 0:
            discrete_arr = np.array([p.discrete for p in pos], dtype=np.bool_)
            if discrete_arr.ndim == 1:
                discrete_arr = discrete_arr.reshape(population_size, self.d_dim)
            acc_d = g_field(
                population_size=population_size,
                dim=self.d_dim,
                pos=discrete_arr,
                mass=mass,
                current_iter=current_iter,
                max_iters=max_iters,
                gravity_constant=gravity_constant.discrete,
                r_power=r_power,
                elitist_check=elitist_check,
                real=False
            )
        else:
            acc_d = []

        acceleration = []
        for i in range(population_size):
            r_acc = acc_r[i] if self.r_dim > 0 else None
            d_acc = acc_d[i] if self.d_dim > 0 else None
            acceleration.append(Acceleration(real=r_acc, discrete=d_acc))

        return acceleration

    def _clip_positions(self, solution: Solution) -> Solution:
        if self.r_dim > 0:
            l1_r = []
            for i, val in enumerate(solution.real):
                l1_r.append(np.clip(val, self.boundaries.real[i][0], self.boundaries.real[i][1]))
        else:
            l1_r = np.array([])

        if self.d_dim > 0:
            discrete_bounds = np.array(self.boundaries.discrete)
            l1_d = np.clip(
                solution.discrete,
                discrete_bounds[:, 0],
                discrete_bounds[:, 1]
            ).astype(int)
        else:
            l1_d = np.array([])

        return Solution(real=np.array(l1_r, dtype=float), discrete=np.array(l1_d, dtype=float))

    def _move(
        self,
        position: List[Solution],
        velocity: List[Velocity],
        acceleration: List[Acceleration],
        population: int = 1,
        v_max: int = 6,
        repair_solution: bool = False
    ) -> Tuple[List[Solution], List[Velocity]]:
        for i in range(population):
            if self.r_dim > 0:
                # Ensure real parts are arrays
                pos_real = np.asarray(position[i].real)
                acc_real = np.asarray(acceleration[i].real)
                vel_real = np.asarray(velocity[i].real)
                
                r1 = np.random.random(pos_real.shape)
                new_vel_real = vel_real * r1 + acc_real
                new_pos_real = np.round(pos_real + new_vel_real)
                
                velocity[i].real = new_vel_real
                position[i].real = new_pos_real

            if self.d_dim > 0:
                r2 = np.random.random(position[i].discrete.shape)
                velocity[i].discrete = velocity[i].discrete * r2 + acceleration[i].discrete
                velocity[i].discrete = np.clip(velocity[i].discrete, a_min=None, a_max=v_max)

                discrete_move_probs = np.abs(np.tanh(velocity[i].discrete))
                rand = np.random.rand(*discrete_move_probs.shape)

                position[i].discrete[rand < discrete_move_probs] = 1 - position[i].discrete[rand < discrete_move_probs]
                position[i].discrete = position[i].discrete.astype(int)

                if not np.any(position[i].discrete):
                    max_index = np.argmax(position[i].discrete)
                    position[i].discrete[max_index] = 1

            new_solution = Solution(position[i].real, discrete=position[i].discrete)

            if not self.is_feasible(new_solution):
                if repair_solution:
                    new_solution = self.custom_repair(new_solution)
                else:
                    new_solution = self._clip_positions(solution=new_solution)

            position[i] = new_solution

        return position, velocity

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ['PYTHONHASHSEED'] = str(seed)