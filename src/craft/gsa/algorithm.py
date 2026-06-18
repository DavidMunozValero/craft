"""Gravitational Search Algorithm (GSA) for railway timetabling.

Custom hybrid GSA handling both real (departure times) and discrete
(scheduled-services mask) decision variables. The dynamic elements and the
gravitational-field primitives live in :mod:`craft.gsa.elements` and
:mod:`craft.gsa.fields` respectively; the shared ``Boundaries``/``Solution``
containers come from :mod:`craft.common`.
"""

import os
import random
import time
from typing import Callable, List, Union

import numpy as np
import pandas as pd

from ..common import Boundaries, Solution
from .elements import Acceleration, GConstant, Velocity
from .fields import (
    g_bin_constant,
    g_field,
    g_real_constant,
    mass_calculation,
    sin_chaotic_term,
)


class GSA:
    def __init__(
        self,
        objective_function: Callable,
        r_dim: int,
        d_dim: int,
        boundaries: Boundaries,
        is_feasible: Union[Callable, None] = None,
        custom_repair: Union[None, Callable] = None,
    ) -> None:
        self.objective_function = objective_function
        self.is_feasible = is_feasible if is_feasible is not None else (lambda _: True)
        self.custom_repair = custom_repair
        self.r_dim = r_dim
        self.d_dim = d_dim
        self.t_dim = self.r_dim + self.d_dim
        self.boundaries = boundaries

        self.population_history = pd.DataFrame()
        self.objective_function_name = self.objective_function.__name__
        self.solution_history: Union[List[Solution], None] = None
        self.accuracy_history: Union[list, None] = None
        self.convergence: Union[np.ndarray, None] = None
        self.start_time: Union[str, None] = None
        self.end_time: Union[str, None] = None
        self.execution_time: Union[float, None] = None
        self.best_solution: Union[Solution, None] = None
        self.best_fitness: float = float("-inf")

    def _get_initial_positions(self, population_size: int) -> List[Solution]:
        pos_r = np.array(
            [
                np.random.uniform(low=rd_lb, high=rd_ub, size=population_size)
                for rd_lb, rd_ub in self.boundaries.real
            ]
        ).T

        pos_d = np.array(
            [
                np.random.choice(a=range(dd_lb, dd_ub + 1), size=population_size)
                for dd_lb, dd_ub in self.boundaries.discrete
            ]
        ).T

        population = []
        for sol in range(population_size):
            real_part = pos_r[sol, :] if self.r_dim > 0 else np.array([])
            discrete_part = pos_d[sol, :] if self.d_dim > 0 else np.array([])

            iters = 0
            while (
                not self.is_feasible(Solution(real=real_part, discrete=discrete_part))
                and iters < 100
            ):
                if self.r_dim > 0:
                    real_part = np.array(
                        [
                            np.random.uniform(low=rd_lb, high=rd_ub)
                            for rd_lb, rd_ub in self.boundaries.real
                        ]
                    )
                if self.d_dim > 0:
                    discrete_part = np.array(
                        [
                            np.random.choice(a=range(dd_lb, dd_ub + 1))
                            for dd_lb, dd_ub in self.boundaries.discrete
                        ]
                    )
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
        seed: Union[int, None] = None,
        verbose: bool = True,
        callback: Union[Callable, None] = None,
    ) -> pd.DataFrame:
        if seed is not None:
            self.set_seed(seed)

        vel = [
            Velocity(
                np.zeros(self.r_dim, dtype=np.float64),
                np.zeros(self.d_dim, dtype=np.float64),
            )
            for _ in range(population_size)
        ]
        fit = np.zeros(population_size, dtype=np.float64)
        accs = np.zeros(population_size, dtype=np.float64)
        mass = np.zeros(population_size, dtype=np.float64)

        g_best = Solution(
            real=np.zeros(self.r_dim, dtype=np.float64),
            discrete=np.zeros(self.d_dim, dtype=np.int64),
        )
        g_best_score = float("-inf")
        g_best_acc = 0.0

        if initial_population is None:
            pos = self._get_initial_positions(population_size)
        else:
            pos = initial_population

        best_solution_history = []
        convergence_curve = np.zeros(iters)

        timer_start = time.time()
        self.start_time = time.strftime("%Y-%m-%d-%H-%M-%S")

        columns = [
            "Iteration",
            "Fitness",
            "Accuracy",
            "ExecutionTime",
            "Discrete",
            "Real",
        ]
        history = pd.DataFrame(columns=columns)

        if verbose:
            print(f"GSA starting: {population_size} agents, {iters} iterations")

        for current_iter in range(iters):
            for i in range(population_size):
                fitness, accuracy = self.objective_function(pos[i])
                fit[i] = fitness
                accs[i] = accuracy

            best_idx = int(np.argmax(fit))
            if fit[best_idx] > g_best_score:
                g_best_score = float(fit[best_idx])
                g_best_acc = float(accs[best_idx])
                g_best = Solution(
                    real=np.array(pos[best_idx].real, dtype=np.float64),
                    discrete=np.array(pos[best_idx].discrete, dtype=np.int64),
                )

            convergence_curve[current_iter] = g_best_score
            best_solution_history.append(
                Solution(
                    real=np.array(g_best.real, dtype=np.float64),
                    discrete=np.array(g_best.discrete, dtype=np.int64),
                )
            )
            history.loc[current_iter] = [
                current_iter + 1,
                g_best_score,
                g_best_acc,
                time.time() - timer_start,
                g_best.discrete,
                g_best.real,
            ]

            if verbose:
                print(
                    f"At iteration {current_iter + 1} the best fitness is {g_best_score}"
                )

            mass = mass_calculation(fit=fit)

            gravity_constant = self._calculate_gravitational_constants(
                current_iter=current_iter,
                max_iters=iters,
                chaotic_constant=chaotic_constant,
                w_max=w_max,
                w_min=w_min,
            )

            acc = self._calculate_acceleration(
                population_size=population_size,
                pos=pos,
                mass=mass,
                current_iter=current_iter,
                max_iters=iters,
                gravity_constant=gravity_constant,
                r_power=r_power,
                elitist_check=elitist_check,
            )

            pos, vel = self._move(
                position=pos,
                velocity=vel,
                acceleration=acc,
                population=population_size,
                v_max=6,
                repair_solution=repair_solution,
            )

        timer_end = time.time()
        self.end_time = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.execution_time = timer_end - timer_start
        self.convergence = convergence_curve
        self.solution_history = best_solution_history
        self.best_solution = g_best
        self.best_fitness = g_best_score

        if verbose:
            print(f"GSA finished: best fitness = {g_best_score}")

        return history

    @staticmethod
    def _calculate_gravitational_constants(
        current_iter: int,
        max_iters: int,
        chaotic_constant: bool,
        w_max: float,
        w_min: float,
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
        elitist_check: bool = True,
    ) -> List[Acceleration]:
        real_arr = np.array([p.real for p in pos], dtype=float)

        if real_arr.ndim == 1:
            if self.r_dim > 0 and real_arr.shape[0] == population_size * self.r_dim:
                real_arr = real_arr.reshape(population_size, self.r_dim)
            elif real_arr.shape[0] == self.r_dim:
                real_arr = real_arr.reshape(1, self.r_dim)

        acc_r: np.ndarray
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
                real=True,
            )
        else:
            acc_r = np.array([])

        acc_d: np.ndarray
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
                real=False,
            )
        else:
            acc_d = np.array([])

        acceleration: List[Acceleration] = []
        for i in range(population_size):
            r_acc = acc_r[i] if self.r_dim > 0 else None
            d_acc = acc_d[i] if self.d_dim > 0 else None
            acceleration.append(Acceleration(real=r_acc, discrete=d_acc))

        return acceleration

    def _clip_positions(self, solution: Solution) -> Solution:
        l1_r: np.ndarray
        if self.r_dim > 0:
            l1_r = np.array(
                [
                    np.clip(val, self.boundaries.real[i][0], self.boundaries.real[i][1])
                    for i, val in enumerate(solution.real)
                ]
            )
        else:
            l1_r = np.array([])

        l1_d: np.ndarray
        if self.d_dim > 0:
            discrete_bounds = np.array(self.boundaries.discrete)
            l1_d = np.clip(
                solution.discrete, discrete_bounds[:, 0], discrete_bounds[:, 1]
            ).astype(int)
        else:
            l1_d = np.array([])

        return Solution(
            real=np.array(l1_r, dtype=float), discrete=np.array(l1_d, dtype=int)
        )

    def _move(
        self,
        position: List[Solution],
        velocity: List[Velocity],
        acceleration: List[Acceleration],
        population: int = 1,
        v_max: int = 6,
        repair_solution: bool = False,
    ):
        for i in range(population):
            if self.r_dim > 0:
                pos_real = np.asarray(position[i].real)
                acc_real = np.asarray(acceleration[i].real)
                vel_real = np.asarray(velocity[i].real)

                r1 = np.random.random(pos_real.shape)
                new_vel_real = vel_real * r1 + acc_real
                new_pos_real = pos_real + new_vel_real

                velocity[i].real = new_vel_real
                position[i].real = new_pos_real

            if self.d_dim > 0:
                r2 = np.random.random(position[i].discrete.shape)
                velocity[i].discrete = (
                    velocity[i].discrete * r2 + acceleration[i].discrete
                )
                velocity[i].discrete = np.clip(
                    velocity[i].discrete, a_min=None, a_max=v_max
                )

                discrete_move_probs = np.abs(np.tanh(velocity[i].discrete))
                rand = np.random.rand(*discrete_move_probs.shape)

                position[i].discrete[rand < discrete_move_probs] = (
                    1 - position[i].discrete[rand < discrete_move_probs]
                )
                position[i].discrete = position[i].discrete.astype(int)

                if not np.any(position[i].discrete):
                    max_index = np.argmax(position[i].discrete)
                    position[i].discrete[max_index] = 1

            new_solution = Solution(position[i].real, discrete=position[i].discrete)

            if not self.is_feasible(new_solution):
                if repair_solution and self.custom_repair is not None:
                    new_solution = self.custom_repair(new_solution)
                else:
                    new_solution = self._clip_positions(solution=new_solution)

            position[i] = new_solution

        return position, velocity

    @staticmethod
    def set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        os.environ["PYTHONHASHSEED"] = str(seed)
