# Optimizers

CRAFT provides three metaheuristic backends for the timetabling problem: the
custom GSA and the mealpy algorithms optimize the real-valued departure times
and share the same conflict-avoiding heuristic for the discrete scheduled
mask, while the GA backend uses a direct genome that also decides which
services are planned.

## GSA (Gravitational Search Algorithm)

A custom hybrid implementation in `craft.gsa`, based on the original GSA_M
project. Located in:

| Module | Contents |
|---|---|
| `craft.gsa.algorithm` | The `GSA` class — `optimize()`, `_move()`, `_calculate_acceleration()`, `_clip_positions()`. |
| `craft.gsa.elements` | `Velocity`, `Acceleration`, `GConstant` containers (real + discrete parts). |
| `craft.gsa.fields` | `mass_calculation`, `g_bin_constant`, `g_real_constant`, `sin_chaotic_term`, `g_field`. |

### How it works

1. **Initialization**: Random positions within bounds; infeasible solutions
   are re-sampled (up to 100 retries).
2. **Fitness evaluation**: The objective function returns a
   `(fitness, accuracy)` tuple.
3. **Mass calculation**: Fitnesses are normalized to masses in `[0, 1]`.
4. **Gravitational constant**: Decreases over iterations (exponentially for
   the real space, linearly for the discrete space). An optional chaotic term
   perturbs it.
5. **Acceleration**: The `k_best` agents (heaviest masses) attract the rest.
   Distance is Euclidean (real) or Hamming (discrete).
6. **Movement**: Velocity is updated and positions are moved. Real positions
   are continuous; discrete positions flip bits probabilistically via a
   tanh-gated velocity.
7. **Best tracking**: `g_best` is updated each iteration; `history`,
   `convergence`, and `solution_history` are populated.

### Key features

- **Reproducible**: Pass `seed=...` to `optimize()`.
- **Verbose gating**: `verbose=True` prints per-iteration progress;
  `verbose=False` is silent.
- **Outputs**: `gsa.best_solution`, `gsa.best_fitness`, `gsa.convergence`,
  `gsa.solution_history`, and the returned `history` DataFrame.

### Usage

```python
from craft.gsa import GSA, Boundaries

def objective(solution):
    # ... compute fitness ...
    return fitness, accuracy

bounds = Boundaries(real=[(lb, ub) for ...], discrete=[])
gsa = GSA(objective_function=objective, r_dim=n_real, d_dim=0, boundaries=bounds)
history = gsa.optimize(population_size=20, iters=50, seed=42, verbose=False)
print(gsa.best_fitness)
```

## GA (direct encoding)

A genetic algorithm over a **direct genome** that encodes the full decision:
for every service, whether it is planned (`b`) and the integer minute deltas
(`δ`) with respect to the requested timetable. Deltas accumulate downstream
when decoded, so the IM margin, operating times and dwell caps hold by
construction; the only remaining constraint (inter-service conflicts) is
handled as a fitness penalty over a **dynamically recomputed** conflict
matrix.

| Module | Contents |
|---|---|
| `craft.ga.base` | The `Problem` / `DiagnosableProblem` protocols — the single integration seam between problem heads and solvers. |
| `craft.ga.timetabling` | `GATimetabling` (the problem head) + `load_problem` (loads the notebook-0.0 instance), decoding, dynamic conflict checking (`CorridorConflicts`), penalized-revenue fitness and reporting helpers (`describe_genome`, `schedule_frame`, `greedy_baseline`). |
| `craft.ga.algorithm` | `GA` — tournament selection, block-wise crossover, integer gaussian mutation with reset-to-requested events, elitism, convergence history and best-genome snapshots. |

> Full formulation, equations mapped to the code and Mermaid diagrams of the
> data flow: `docs/approach_ga_direct_encoding.md` (Spanish, follows the
> notation of the TimetablingProblemRailway paper).
| `craft.ga.plotting` | Genome heatmap and filmstrip, Marey views (`plot_timetable`, `plot_timetable_filmstrip`) and the run summary (`plot_evolution`). |

### Plugging other solvers

`GA` consumes any object implementing `DiagnosableProblem` (a superset of the
minimal `Problem` contract: genome size, per-gene bounds, service blocks and
the `fitness`/`decode` callbacks). A custom GA only has to implement that
contract to work over `GATimetabling` unchanged:

```python
problem = load_problem("data/synthetic/timetable.yaml", "data/synthetic/revenue.yaml")
my_solver = MyCustomGA(problem, ...)
history = my_solver.optimize(epoch=..., seed=...)
solution = problem.decode(my_solver.best_genome)
```

### Fitness

```
fitness = Σ_{planned} revenue_i(decoded schedule)
          − λ · Σ_{conflicting planned pairs} min(revenue_i, revenue_j)
```

The default `λ = 2` makes every conflict strictly worse than keeping only the
more profitable service of the pair, so the search is always rewarded for
resolving conflicts.

### Usage

```python
from craft.ga import GA, load_problem

problem = load_problem("data/synthetic/timetable.yaml", "data/synthetic/revenue.yaml")
ga = GA(problem, pop_size=40)
history = ga.optimize(epoch=100, seed=42)
print(ga.best_fitness, ga.best_solution.discrete.sum())
```

The input files are generated by notebook `0.0-Synthetic_Data_Generator`.
`scripts/run_ga.py` wraps the whole flow (load instance → optimize → save
convergence CSV, best-schedule CSV and evolution filmstrips under
`data/results/`).

## mealpy backends

CRAFT wraps the [mealpy](https://github.com/thieu199h/mealpy) library to
solve the same problem with several established metaheuristics. The problem
formulation lives in `craft.mealpy.timetabling.MealpyTimetabling`.

### Available algorithms

| Name | mealpy class | Constructor |
|---|---|---|
| GA | `GA.BaseGA` | `GA.BaseGA(epoch, pop_size, pc, pm)` |
| DE | `DE.OriginalDE` | `DE.OriginalDE(epoch, pop_size, wf, cr)` |
| PSO | `PSO.OriginalPSO` | `PSO.OriginalPSO(epoch, pop_size, c1, c2, w)` |
| SCA | `SCA.OriginalSCA` | `SCA.OriginalSCA(epoch, pop_size)` |

> **mealpy 3.x API**: `epoch` and `pop_size` go in the constructor; `seed`
> goes in `solve(problem, seed=...)`. The correct class names are
> `DE.OriginalDE` and `SCA.OriginalSCA` (not `DE.Original` / `SCA.Original`).

### Constraints

- `pop_size >= 10` is required for GA (k-way tournament selection needs
  enough candidates).

### Usage

```python
from mealpy import FloatVar, GA
from craft.mealpy import MealpyTimetabling

tt = MealpyTimetabling(services, revenue_behavior, safe_headway=10, max_stop_time=10)
bounds = [FloatVar(lb=lb, ub=ub) for lb, ub in tt.boundaries.real]
problem = {"obj_func": tt.objective_function, "bounds": bounds, "minmax": "max", "verbose": False}
model = GA.BaseGA(epoch=50, pop_size=20, pc=0.9, pm=0.01)
model.solve(problem, seed=42)
print(model.g_best.target.fitness)
```

## Fairness-aware variant

The fairness-aware optimization (`scripts/run_fairness.py`,
`ExperimentRunner._optimize_fairness`) uses a GA with a combined objective:

```python
def fairness_objective(solution):
    revenue = tt.objective_function(solution)
    schedule = tt.get_heuristic_schedule()
    fair_idx, _ = FairnessMetrics.jains_fairness_index(schedule, tt.capacities, revenue_behavior)
    return alpha * revenue / 1e3 + beta * fair_idx * 100
```

Tune the revenue/fairness trade-off with `--alpha` and `--beta`.

## Adding a new algorithm

### Adding a mealpy algorithm

1. Add the algorithm to the registry in `craft.runner.build_mealpy_algorithm`:

```python
from mealpy import ...  # your algorithm module

algorithms = {
    ...
    "your_algo": lambda: YourAlgo(epoch=epoch, pop_size=pop_size, ...),
}
```

2. Add the choice to `scripts/run_mealpy.py`'s argparse `choices=[...]`.
3. Add a test case in `tests/test_runner.py` (`test_build_known_algorithms`).

### Adding a custom backend

1. Implement the optimizer as a class with an `optimize()` method (or
   compatible interface) in a new module under `craft/`.
2. Add a dispatch branch in `ExperimentRunner.optimize()`.
3. Add a CLI script in `scripts/` (following the `run_gsa.py` pattern).
4. Add tests in `tests/`.
