"""Experiment runner for CRAFT timetabling optimization.

Encapsulates the common pipeline shared by all experiments:

    generate supply -> load -> revenue behavior -> timetabling problem
    -> optimize (GSA or mealpy) -> update supply -> save -> plot

Configurable via :class:`ExperimentConfig` (dataclass) so that the thin
CLI scripts in ``scripts/`` only need to parse arguments and call
:meth:`ExperimentRunner.run`.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import numpy as np
import pandas as pd

from robin.supply.entities import Supply
from robin.supply.generator.entities import SupplyGenerator
from robin.supply.saver.entities import SupplySaver

from .common import Boundaries, Solution
from .fairness import FairnessMetrics
from .mealpy import MealpyTimetabling
from .revenue import RevenueSimulator

DEFAULT_SUPPLY_CONFIG = Path("configs/supply_generator/supply_data.yaml")
DEFAULT_GENERATOR_CONFIG = Path("configs/supply_generator/config.yaml")


@dataclass
class ExperimentConfig:
    """Configuration for a single optimization experiment.

    Paths are resolved relative to the project root (the directory containing
    ``configs/`` and ``src/``) when the runner is invoked from a script.
    """

    # Supply generation
    supply_config_path: Path = DEFAULT_SUPPLY_CONFIG
    generator_config_path: Path = DEFAULT_GENERATOR_CONFIG
    output_supply_path: Union[Path, None] = None
    n_services: int = 25
    without_conflicts: bool = False

    # Revenue simulation
    revenue_alpha: float = 2 / 3

    # Timetabling problem
    safe_headway: int = 10
    max_stop_time: int = 10

    # Optimization
    algorithm: str = "ga"
    seed: int = 42
    pop_size: int = 20
    epoch: int = 50

    # Fairness (only used when algorithm == "fairness")
    fair_index: Union[str, None] = None
    fairness_alpha: float = 0.7
    fairness_beta: float = 0.3

    # Output
    results_dir: Path = Path("data/results")
    figures_dir: Path = Path("reports/figures")
    save_convergence: bool = True
    verbose: bool = True

    def resolved_paths(self, project_root: Path) -> dict:
        """Resolve relative paths against ``project_root``."""

        def _resolve(p: Path) -> Path:
            return p if p.is_absolute() else project_root / p

        return {
            "supply_config": _resolve(self.supply_config_path),
            "generator_config": _resolve(self.generator_config_path),
            "results_dir": _resolve(self.results_dir),
            "figures_dir": _resolve(self.figures_dir),
            "output_supply": _resolve(self.output_supply_path)
            if self.output_supply_path
            else None,
        }


def build_mealpy_algorithm(name: str, epoch: int, pop_size: int):
    """Build a mealpy optimizer instance by name.

    Args:
        name: One of ``"ga"``, ``"de"``, ``"pso"``, ``"sca"``.
        epoch: Number of epochs (iterations).
        pop_size: Population size.

    Returns:
        A mealpy optimizer instance (not yet solved).

    Raises:
        ValueError: If ``name`` is not a recognized algorithm.
    """
    from mealpy import GA, DE, PSO, SCA

    algorithms = {
        "ga": lambda: GA.BaseGA(epoch=epoch, pop_size=pop_size, pc=0.9, pm=0.01),
        "de": lambda: DE.OriginalDE(epoch=epoch, pop_size=pop_size, wf=0.5, cr=0.9),
        "pso": lambda: PSO.OriginalPSO(
            epoch=epoch, pop_size=pop_size, c1=1.5, c2=1.5, w=0.7
        ),
        "sca": lambda: SCA.OriginalSCA(epoch=epoch, pop_size=pop_size),
    }
    key = name.lower()
    if key not in algorithms:
        raise ValueError(f"Unknown algorithm '{name}'. Choose from {list(algorithms)}")
    return algorithms[key]()


class ExperimentRunner:
    """Run a complete timetabling optimization experiment.

    The runner executes the full pipeline (generate -> load -> revenue ->
    timetabling -> optimize -> save) and returns a results dictionary with the
    best fitness, scheduled-services count, convergence curve and paths to the
    saved artifacts.
    """

    def __init__(
        self, config: ExperimentConfig, project_root: Union[Path, None] = None
    ) -> None:
        self.config = config
        self.project_root = project_root or Path.cwd()
        self.paths = config.resolved_paths(self.project_root)

        self.supply: Union[Supply, None] = None
        self.timetabling: Union[MealpyTimetabling, None] = None
        self.convergence: Union[np.ndarray, None] = None
        self.best_solution: Union[Solution, None] = None

    def generate_supply(self) -> Path:
        """Generate a supply YAML and return its path."""
        paths = self.paths
        if paths["output_supply"] is None:
            paths["output_supply"] = (
                paths["results_dir"]
                / f"supply_{self.config.algorithm}_seed{self.config.seed}.yaml"
            )

        paths["results_dir"].mkdir(parents=True, exist_ok=True)

        generator = SupplyGenerator.from_yaml(
            path_config_supply=paths["supply_config"],
            path_config_generator=paths["generator_config"],
        )
        generator.generate(
            n_services=self.config.n_services,
            output_path=paths["output_supply"],
            seed=self.config.seed,
            progress_bar=self.config.verbose,
            without_conflicts=self.config.without_conflicts,
        )
        if self.config.verbose:
            print(
                f"Generated {len(generator.services)} services -> {paths['output_supply']}"
            )
        return Path(paths["output_supply"])

    def load_supply(self, supply_path: Union[Path, None] = None) -> Supply:
        """Load a supply YAML file."""
        path = supply_path or self.paths["output_supply"]
        if path is None:
            raise ValueError(
                "No supply path provided. Either generate one or pass --supply."
            )
        path = Path(path)
        self.supply = Supply.from_yaml(path=str(path))
        self.paths["output_supply"] = path
        if self.config.verbose:
            print(f"Loaded {len(self.supply.services)} services from {path}")
        return self.supply

    def build_timetabling(
        self, supply: Union[Supply, None] = None
    ) -> MealpyTimetabling:
        """Build the timetabling problem from the supply and revenue behavior."""
        supply = supply or self.supply
        assert supply is not None
        revenue_behavior = RevenueSimulator(supply=supply).simulate_revenue(
            alpha=self.config.revenue_alpha,
        )
        self.timetabling = MealpyTimetabling(
            requested_services=supply.services,
            revenue_behavior=revenue_behavior,
            safe_headway=self.config.safe_headway,
            max_stop_time=self.config.max_stop_time,
        )
        if self.config.verbose:
            n_real = len(self.timetabling.boundaries.real)
            print(
                f"Timetabling problem: {self.timetabling.n_services} services, {n_real} real variables"
            )
        return self.timetabling

    def optimize(self) -> dict:
        """Run the optimizer and return a results dictionary.

        Dispatches to :meth:`_optimize_mealpy` or :meth:`_optimize_gsa`
        depending on ``config.algorithm``.
        """
        if self.timetabling is None:
            self.build_timetabling()
        assert self.timetabling is not None

        if self.config.algorithm == "gsa":
            return self._optimize_gsa()
        elif self.config.algorithm == "fairness":
            return self._optimize_fairness()
        else:
            return self._optimize_mealpy()

    def _optimize_mealpy(self) -> dict:
        from mealpy import FloatVar

        assert self.timetabling is not None
        tt = self.timetabling
        bounds = [FloatVar(lb=lb, ub=ub) for lb, ub in tt.boundaries.real]
        problem = {
            "obj_func": tt.objective_function,
            "bounds": bounds,
            "minmax": "max",
            "verbose": False,
        }

        model = build_mealpy_algorithm(
            self.config.algorithm, self.config.epoch, self.config.pop_size
        )
        if self.config.verbose:
            print(
                f"Running {self.config.algorithm.upper()} (epoch={self.config.epoch}, pop={self.config.pop_size}, seed={self.config.seed})..."
            )
        model.solve(problem, seed=self.config.seed)

        best_position = model.g_best.solution
        best_fitness = float(model.g_best.target.fitness)
        convergence = np.array(
            [d.target.fitness for d in model.history.list_global_best]
        )

        schedule = tt.get_heuristic_schedule()
        self.best_solution = Solution(real=best_position, discrete=schedule)
        self.convergence = convergence

        n_scheduled = int(np.sum(schedule))
        if self.config.verbose:
            print(
                f"Best fitness: {best_fitness:.2f}, scheduled: {n_scheduled}/{len(schedule)}"
            )

        return {
            "algorithm": self.config.algorithm,
            "seed": self.config.seed,
            "fitness": best_fitness,
            "n_scheduled": n_scheduled,
            "n_total": len(schedule),
            "convergence": convergence,
        }

    def _optimize_gsa(self) -> dict:
        from .gsa import GSA

        assert self.timetabling is not None
        tt = self.timetabling
        n_real = len(tt.boundaries.real)
        boundaries = Boundaries(real=tt.boundaries.real, discrete=[])

        def gsa_objective(sol):
            sol_arr = np.array(sol.real, dtype=np.int32)
            tt.schedule_manager.update_from_solution(sol_arr)
            tt.revenue_calculator.updated_schedule = (
                tt.schedule_manager.updated_schedule
            )
            tt.revenue_calculator.recompute_all_revenues()
            schedule = tt.get_heuristic_schedule()
            revenue = tt.get_revenue(Solution(real=sol.real, discrete=schedule))
            accuracy = float(np.sum(schedule)) / len(schedule)
            return revenue, accuracy

        gsa = GSA(
            objective_function=gsa_objective,
            r_dim=n_real,
            d_dim=0,
            boundaries=boundaries,
        )
        if self.config.verbose:
            print(
                f"Running GSA (iters={self.config.epoch}, pop={self.config.pop_size}, seed={self.config.seed})..."
            )
        gsa.optimize(
            population_size=self.config.pop_size,
            iters=self.config.epoch,
            seed=self.config.seed,
            verbose=False,
        )

        convergence = gsa.convergence
        assert gsa.best_solution is not None
        schedule = tt.get_heuristic_schedule()
        self.best_solution = Solution(real=gsa.best_solution.real, discrete=schedule)
        self.convergence = convergence

        n_scheduled = int(np.sum(schedule))
        best_fitness = gsa.best_fitness
        if self.config.verbose:
            print(
                f"Best fitness: {best_fitness:.2f}, scheduled: {n_scheduled}/{len(schedule)}"
            )

        return {
            "algorithm": "gsa",
            "seed": self.config.seed,
            "fitness": float(best_fitness),
            "n_scheduled": n_scheduled,
            "n_total": len(schedule),
            "convergence": convergence,
        }

    def _optimize_fairness(self) -> dict:
        from mealpy import FloatVar, GA

        assert self.timetabling is not None
        tt = self.timetabling
        revenue_behavior = tt.revenue
        alpha = self.config.fairness_alpha
        beta = self.config.fairness_beta

        def fairness_objective(solution):
            revenue = tt.objective_function(solution)
            schedule = tt.get_heuristic_schedule()
            fair_idx, _ = FairnessMetrics.jains_fairness_index(
                schedule,
                tt.capacities,
                revenue_behavior,
            )
            return alpha * revenue / 1e3 + beta * fair_idx * 100

        bounds = [FloatVar(lb=lb, ub=ub) for lb, ub in tt.boundaries.real]
        problem = {
            "obj_func": fairness_objective,
            "bounds": bounds,
            "minmax": "max",
            "verbose": False,
        }

        model = GA.BaseGA(
            epoch=self.config.epoch, pop_size=self.config.pop_size, pc=0.9, pm=0.01
        )
        if self.config.verbose:
            print(
                f"Running fairness-aware GA (epoch={self.config.epoch}, pop={self.config.pop_size}, seed={self.config.seed}, alpha={alpha}, beta={beta})..."
            )
        model.solve(problem, seed=self.config.seed)

        best_position = model.g_best.solution
        best_fitness = float(model.g_best.target.fitness)
        convergence = np.array(
            [d.target.fitness for d in model.history.list_global_best]
        )

        schedule = tt.get_heuristic_schedule()
        self.best_solution = Solution(real=best_position, discrete=schedule)
        self.convergence = convergence

        n_scheduled = int(np.sum(schedule))
        if self.config.verbose:
            print(
                f"Best fitness: {best_fitness:.2f}, scheduled: {n_scheduled}/{len(schedule)}"
            )

        return {
            "algorithm": "fairness",
            "seed": self.config.seed,
            "fitness": best_fitness,
            "n_scheduled": n_scheduled,
            "n_total": len(schedule),
            "convergence": convergence,
        }

    def save_results(self, results: dict) -> dict:
        """Save the updated supply and convergence curve.

        Returns a dictionary with the paths to the saved artifacts.
        """
        paths = self.paths
        assert self.timetabling is not None
        assert self.best_solution is not None
        paths["results_dir"].mkdir(parents=True, exist_ok=True)

        algo = results["algorithm"]
        seed = results["seed"]
        suffix = f"{algo}_seed{seed}"

        updated_services = self.timetabling.update_supply(
            str(paths["output_supply"]),
            self.best_solution,
        )
        supply_path = paths["results_dir"] / f"supply_updated_{suffix}.yaml"
        SupplySaver(updated_services).to_yaml(str(supply_path))

        saved = {"supply_path": supply_path}

        if self.config.save_convergence and self.convergence is not None:
            conv_path = paths["results_dir"] / f"convergence_{suffix}.csv"
            pd.DataFrame(
                {
                    "iteration": range(len(self.convergence)),
                    "fitness": self.convergence,
                }
            ).to_csv(conv_path, index=False)
            saved["convergence_path"] = conv_path

        if self.config.verbose:
            print(f"Saved updated supply -> {supply_path}")
            if "convergence_path" in saved:
                print(f"Saved convergence  -> {saved['convergence_path']}")

        return saved

    def run(self) -> dict:
        """Execute the full pipeline and return the results dictionary."""
        self.generate_supply()
        self.load_supply()
        self.build_timetabling()
        results = self.optimize()
        saved = self.save_results(results)
        results["saved"] = saved
        return results
