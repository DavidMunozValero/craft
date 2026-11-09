# Optimizers

CRAFT provides four metaheuristic backends for the timetabling problem: the
custom GSA and the mealpy algorithms optimize the real-valued departure times
and share the same conflict-avoiding heuristic for the discrete scheduled
mask, while the GA and NSGA-II backends use a direct genome that also decides
which services are planned.

## GSA (Gravitational Search Algorithm)

A custom hybrid implementation in `craft.gsa`, based on the original GSA_M
project. Located in:

| Module | Contents |
|---|---|
| `craft.gsa.algorithm` | The `GSA` class — `optimize()`, `_move()`, `_calculate_acceleration()`, `_clip_positions()`. Supports `custom_repair` + `is_feasible` + `repair_solution=True`. |
| `craft.gsa.elements` | `Velocity`, `Acceleration`, `GConstant` containers (real + discrete parts). |
| `craft.gsa.fields` | `mass_calculation`, `g_bin_constant`, `g_real_constant`, `sin_chaotic_term`, `g_field`. |

### How it works

1. **Initialization**: Random positions within bounds; infeasible solutions
   are re-sampled (up to 100 retries). `ExperimentRunner._optimize_gsa`
   seeds the initial population with the requested timetable plus repaired
   random draws (the GSA_M seeding, restored in P0).
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
7. **Repair (timetabling)**: the runner wires `repair_solution=True` with
   `MealpyTimetabling.repair_departures` — every moved vector is mapped onto
   the schedule the direct encoding would decode (deltas →
   `craft.scheduling.cumulative_shifts`, feasibility by construction). The
   GSA_M repair/seeding was lost in the port and restored in P0; without it
   the GSA collapses on conflicting instances (2.0 ± 0.7 operative services
   on the 50-service instance instead of 17).
8. **Best tracking**: `g_best` is updated each iteration; `history`,
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
| `craft.ga.operators` | `BlockOperators` — the shared block-aware variation operators (seeded initial population, block crossover, gaussian mutation with reset events) used by every genetic backend. |

> Full formulation, equations mapped to the code and Mermaid diagrams of the
> data flow: `docs/04_direct_encoding.md` (Spanish, follows the
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
`data/results/`). For comparative experiments,
`scripts/run_comparison.py` runs the full method × seed grid on one priced
instance with the common evaluator (`craft.ga.evaluation`) and writes
`runs.csv`, `summary.csv`, `indicators_nsga2.csv`, per-run convergences,
measured Pareto fronts and knee schedules under
`data/results/benchmark/<instance tag>/`. Any member of the instance family
(`data/synthetic/instances/<tag>/`, E1) is consumed via
`--timetable/--revenue`.

## NSGA-II (direct encoding)

A self-contained NSGA-II (`craft.ga.nsga2`, no pymoo) over the **same genome,
contract and operators as the GA**: it only replaces the selection and
replacement steps with the classic Deb et al. (2002) machinery. Notebook
`7.0-NSGA2_Direct_Encoding` runs it on the synthetic instance and compares it
against the GA at equal budget.

| Piece | Implementation |
|---|---|
| Fast non-dominated sort | `fast_non_dominated_sort` — vectorized (N, N, M) dominance matrix, fronts peeled by domination counts. |
| Crowding distance | `crowding_distance` — infinite at the boundaries, normalized neighbour gaps inside; degenerate (zero-span) objectives are skipped. |
| Mating selection | Binary tournaments with the crowded comparison operator (lower rank wins; ties go to the less crowded). |
| Environmental selection | μ+λ: parents and offspring are merged and survivors are taken front by front; the front crossing the budget is truncated by descending crowding. |
| Objectives | `objectives` callable, defaulting to the single penalized-revenue fitness of the problem head (exactly the GA's landscape). |
| Hypervolume | `hypervolume` — exact 2-objective area between a front and a reference point (maximization). |

With one objective the fronts collapse to singletons, so NSGA-II degenerates
to an elitist rank-based GA over the merged population — an apples-to-apples
baseline against `GA`. The genuinely multi-objective formulation comes from
`craft.ga.objectives.revenue_fairness_objectives`, which builds a
`[penalized revenue, fairness]` vector per genome: the fairness axis is a
capacity-aware index (Jain/Gini/Atkinson on RU importance-vs-capacity ratios
raised to a sensitivity exponent `alpha`, following the Fair-oriented paper)
computed over the planning mask, so it only depends on *which* services run.
The empty schedule scores `(0, 0)` so the null plan never enters the front.

### Usage

```python
from craft.ga import NSGA2, load_problem, revenue_fairness_objectives

problem = load_problem("data/synthetic/timetable.yaml", "data/synthetic/revenue.yaml")
nsga = NSGA2(
    problem,
    pop_size=40,
    objectives=revenue_fairness_objectives(problem, index="jain", alpha=25.0),
)
history = nsga.optimize(epoch=100, seed=42)
print(nsga.pareto_values.shape)
```

`pop_size >= 10` is validated as in the GA backend. The solver mirrors the GA
outputs (`best_genome`, `best_fitness`, `best_solution`, `best_genomes`,
`convergence`, `execution_time`, history DataFrame) and adds the final
`pareto_genomes` / `pareto_values` (front 0 sorted by descending first
objective) plus `pareto_history` (the first-front objective matrix at every
generation — the raw material for hypervolume curves via `hypervolume`).

The exponents are pre-calibrated as in the Fair-oriented paper
(`craft.ga.objectives.FAIRNESS_ALPHA`: 25 for Jain and Atkinson, 10 for
Gini); passing `alpha=None` picks the calibration of the chosen index.

`scripts/run_nsga2.py` wraps the whole bi-objective flow (load instance →
optimize → save convergence CSV, Pareto-front CSV, knee-point schedule CSV,
hypervolume CSV and the Pareto scatter / hypervolume figures under
`data/results/`). The knee is the front point closest to the ideal corner
after min-max normalization.

### Pareto-front quality indicators

`craft.ga.indicators` is the measurement standard for comparing fronts
produced by different methods (own NSGA-II, external baselines, scalar
sweeps, exact reference points), all under the maximization convention:

| Indicator | Function | Reads |
|---|---|---|
| Normalized hypervolume | `normalized_hypervolume(front, reference)` — min-max normalization against the reference set (e.g. the union of compared fronts), reference point at the nadir corner; scale-free. |
| Inverted generational distance | `igd(front, reference)` — mean Euclidean distance from each reference point to its nearest front point (lower is better; normalize with `min_max_normalize` when objectives have different scales). |
| Spread (Deb's delta) | `spread(front, reference)` — relative uniformity and extent against the reference extremes (lower is better). |
| Set coverage | `c_metric(a, b)` — fraction of `b` weakly dominated by `a` (higher is better). |

`ideal` / `nadir` expose the corners of a front set and
`min_max_normalize(values, low, high)` the shared scaling. The comparison
grid writes them per seed against the union of NSGA-II fronts in
`indicators_nsga2.csv` (all three in scale-free units).

### External baselines (optional `baselines` group)

Two validation harnesses for the publication battery live as top-level
scripts; both need the optional dependency group
(`poetry install --with baselines`) and their tests skip without it.

**pymoo cross-check** (`scripts/run_pymoo.py`): runs the craft NSGA-II and
pymoo's reference NSGA-II over the same genome layout, objectives and
budget. The adapter (`PymooAdapter`) relaxes the integer genome to reals
within the layout bounds, negates the (maximization) objectives — pymoo
minimizes — and rounds genes back to integers at evaluation time; pymoo
brings its own SBX/PM operators, so the comparison isolates the NSGA-II
machinery from the operator design. Both fronts are measured with the common
evaluator and compared with the indicators above (`pymoo_crosscheck.csv`,
per-seed fronts under `data/results/pymoo/`); the validation criterion is
set coverage without massive mutual domination.

**Exact ε-constraint reference** (`scripts/run_epsilon.py`): Pyomo + SCIP
selection-level model over the requested timetables — binary planning mask,
pairwise conflict-freeness over the requested schedules, canon-revenue
objective — restricted by a balance level ``t``: every RU's
achieved-vs-capacity ratio must lie in ``[t·u, u]`` for a common ``u`` (all
RUs participate, no ratio exceeds ``1/t`` times another's). This is a
sufficient condition for the capacity-aware Jain floor ``t**50`` at the
alpha=25 calibration, and the level is *sufficient, exact and nested*: the
MILP per level is solved to optimality and max revenue is non-increasing in
``t``. Restricting the Jain index directly was tried and rejected: the
calibrated index raises ratios to the power 25, so the polynomial constraint
evaluates around 1e-7, below branch-and-bound tolerances — the solver then
declares violated solutions optimal (observed on n25_r3). The linear balance
formulation sidesteps it, and levels that the discrete importance weights
cannot satisfy (e.g. ``t=1``, exact proportionality) are certified
infeasible instead of silently relaxed. Measured points land in
`epsilon_reference.csv` under `data/results/epsilon/<instance tag>/`;
typical read-out on the 25-service instance: the exact conflict-free
selection reaches 457.59 vs the greedy 421.23, and revenue decays
monotonically as the balance tightens.

### Battery orchestration (E1-E4) and statistics (E3)

`scripts/run_battery.py` reads the instance-family manifest and drives the
harnesses per instance as subprocesses: the comparison grid (all methods,
plus the rev*fair sweep of the second paper emitted as one
`ga_mealpy_fair_<index>` method per `--fair-indices` entry, each with its
calibrated alpha), the exact ε-constraint reference (only 25-service
instances, per the plan) and the pymoo cross-check. The NSGA-II itself runs
on the revenue-Jain axis of the paper. Steps whose artifact already exists
are skipped, so a battery run is resumable; `--force` redoes them, and a
failed subprocess is reported at the end with a nonzero exit.

`scripts/run_stats.py` walks the battery directories and writes `stats.csv`
next to every `runs.csv`: one row per (metric, method pair) with the
seed-paired Wilcoxon signed-rank test, the Cliff's delta effect size and its
Romano magnitude band, and the Holm step-down correction within each
(instance, metric) family — the experiment-E3 protocol (20-30 seeds passed
via `--seeds`). The underlying tools live in the shared layer
(`craft.stats`), with known-case tests.

## mealpy backends

CRAFT wraps the [mealpy](https://github.com/thieu199h/mealpy) library to
solve the same problem with several established metaheuristics. The problem
formulation lives in `craft.mealpy.timetabling.MealpyTimetabling`.

### Instance mode

`scripts/run_mealpy.py` and `scripts/run_gsa.py` accept `--timetable` and
`--revenue` to consume the notebook-0.0 instance directly
(`ExperimentRunner.load_instance`), skipping supply generation and revenue
simulation so every backend optimizes the exact same priced instance:

```bash
python scripts/run_mealpy.py -a ga --timetable data/synthetic/timetable.yaml \
    --revenue data/synthetic/revenue.yaml --pop-size 40 --epoch 100
python scripts/run_gsa.py --timetable data/synthetic/timetable.yaml \
    --revenue data/synthetic/revenue.yaml
```

### Repair and determinism

`MealpyTimetabling.repair_departures(real)` maps any departure vector onto
the feasible schedule the direct encoding would decode (cumulative shifts
clipped to the IM margin and the dwell cap via
`craft.scheduling.cumulative_shifts`; already-feasible schedules are fixed
points). The conflict-avoiding heuristic (`get_heuristic_schedule`) breaks
revenue ties by ascending service id, so runs are bit-reproducible across
processes.

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

The **multiplicative** objective of the Fair-oriented paper
(`revenue × fairness`) is available directly on the problem head: build
`MealpyTimetabling(..., fair_index="Jain"|"Gini"|"Atkinson", alpha=25.0)` and
`objective_function` returns `revenue × fairness` (with the calibrated
exponent; `fair_index=None` keeps the pure revenue objective). The empty
schedule scores perfect fairness (1.0) so the null plan never wins the
multiplicative objective with a zero revenue.

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
