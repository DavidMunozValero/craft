# Architecture

This document describes the module structure of CRAFT and the data flow of a
typical optimization experiment.

## Module structure

CRAFT follows a layered design with a shared core and two optimization
backends.

```
                 ┌─────────────────────────────────────────┐
                 │              craft.runner                │
                 │      ExperimentConfig / Runner           │
                 └───────────────┬─────────────────────────┘
                                 │ orchestrates
          ┌──────────────────────┼──────────────────────┐
          ▼                      ▼                      ▼
   ┌─────────────┐      ┌─────────────┐         ┌─────────────┐
   │  craft.gsa  │      │ craft.mealpy│         │  (notebooks │
   │  (backend)  │      │  (backend)  │         │   & scripts)│
   └──────┬──────┘      └──────┬──────┘         └─────────────┘
          │                     │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │    Shared layer      │
          │  common  revenue     │
          │  scheduling fairness │
          │  plotter  utils      │
          └──────────┬──────────┘
                     ▼
          ┌─────────────────────┐
          │     robin            │
          │  (supply, generator, │
          │   saver, plotter)    │
          └─────────────────────┘
```

### Shared layer

| Module | Responsibility |
|---|---|
| `craft.common` | `Boundaries`, `Solution`, `ConflictMatrix` — the data structures shared by both backends. |
| `craft.revenue` | `RevenueSimulator` (synthesizes a revenue behavior per service), `RevenueCalculator` (evaluates revenue of a candidate timetable), `penalty_function`. |
| `craft.scheduling` | `ScheduleManager` (updates schedules from a solution, checks feasibility, computes operational times), `get_schedule_from_supply`. |
| `craft.fairness` | `FairnessMetrics` — Jain, Gini, and Atkinson indices, plus the fairness-index variants that account for RU capacities and service importances. |
| `craft.plotter` | Box/line plots and a scheduled-services chart for experiment analysis. |
| `craft.utils` | `get_rus_revenue`, `is_better_solution` — post-optimization analysis helpers. |

### Backends

| Module | Responsibility |
|---|---|
| `craft.gsa` | Custom hybrid Gravitational Search Algorithm. Split into `algorithm` (the `GSA` class), `elements` (Velocity/Acceleration/GConstant containers), and `fields` (mass calculation, gravitational constants, the `g_field` function). |
| `craft.mealpy` | `MealpyTimetabling` — the problem formulation (objective, boundaries, feasibility, heuristic schedule) solved with the algorithms from the `mealpy` library. |

### Orchestrator

| Module | Responsibility |
|---|---|
| `craft.runner` | `ExperimentConfig` (dataclass) + `ExperimentRunner` — encapsulates the full pipeline so that the CLI scripts in `scripts/` only parse arguments and call `runner.run()`. |

### Public API

`craft/__init__.py` eagerly exports the shared layer and lazily loads the
backends (`craft.GSA`, `craft.MealpyTimetabling`) via `__getattr__`. This
keeps `import craft` light — `scipy.spatial` and the `mealpy` library are
only imported when a backend is actually used.

## Data flow

A complete experiment follows this pipeline:

```
1. Generate supply
   SupplyGenerator (robin) + config YAMLs
   → supply YAML (list of services with lines, stations, timetables, RUs)

2. Load supply
   Supply.from_yaml → Supply object

3. Revenue behavior
   RevenueSimulator(supply).simulate_revenue(alpha)
   → per-service {canon, k, dt_max_penalty, tt_max_penalty, importance, ru}

4. Build timetabling problem
   MealpyTimetabling(services, revenue_behavior, safe_headway, max_stop_time)
   → reference_schedules, operational_times, boundaries (real), conflict_matrix

5. Optimize
   GSA.optimize(...)   or   mealpy model.solve(problem, seed)
   → best_solution (real = departure times)

6. Discrete schedule
   timetabling.get_heuristic_schedule()
   → boolean mask (which services are scheduled, conflict-free)

7. Update & save
   timetabling.update_supply(path, solution) → updated services
   SupplySaver(updated_services).to_yaml(path)
   → supply_updated_{algo}_seed{seed}.yaml

8. Convergence
   runner.save_results()
   → convergence_{algo}_seed{seed}.csv
```

### The heuristic schedule

Both backends optimize only the real-valued departure times. The discrete
part — which services are scheduled — is determined by
`MealpyTimetabling.get_heuristic_schedule()`:

1. Start with all services marked as schedulable.
2. Identify services that conflict (via `ConflictMatrix`).
3. Greedily drop the lowest-revenue conflicting service until no conflicts
   remain.

This makes the GSA and mealpy backends directly comparable: they solve the
same real-valued subproblem and share the same discrete heuristic.

## Configuration

The supply and generator configurations live in `configs/supply_generator/`:

| File | Contents |
|---|---|
| `supply_data.yaml` | Stations, corridors, lines, rolling stocks, TSPs, seats. |
| `config.yaml` | Dates, line/TSP probabilities, time-slot distributions, price factors. |

These are consumed by `robin`'s `SupplyGenerator.from_yaml()`.
