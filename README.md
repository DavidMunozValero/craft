# CRAFT

**Capacity & Revenue-Aware Fair Timetabling** — metaheuristic optimization of
railway service timetables on top of the
[robin](https://github.com/JoseAngelMartinB/robin) railway simulator.

CRAFT formulates the Infrastructure Manager's revenue-maximization timetabling
problem and solves it with two metaheuristic backends:

- **GSA** — a custom hybrid Gravitational Search Algorithm.
- **mealpy** — several algorithms (GA, DE, PSO, SCA) from the
  [mealpy](https://github.com/thieu199h/mealpy) library.

Both backends optimize the real-valued departure times; the discrete part
(which services are scheduled) is obtained with a conflict-avoiding heuristic.
A fairness-aware variant combines revenue with Jain's / Gini's / Atkinson's
indices to balance capacity allocation across Railway Undertakings (RUs).

---

## Installation

CRAFT depends on the `robin` simulator, installed as an editable path
dependency from a sibling checkout of
[JoseAngelMartinB/robin](https://github.com/JoseAngelMartinB/robin) (master
branch).

### Prerequisites

- Python ≥ 3.11
- [Poetry](https://python-poetry.org/) ≥ 2.0
- A local clone of `robin` next to this project:

```bash
cd /path/to/your/projects
git clone https://github.com/JoseAngelMartinB/robin.git
```

### Setup

```bash
cd /path/to/your/projects/craft
poetry lock
poetry install
```

This installs `robin` in editable mode (from `../robin`), all runtime
dependencies, and the development tools (`pytest`, `ruff`, `mypy`, `jupyter`).

> **Note:** The `robin` path dependency is declared in `pyproject.toml` as
> `robin = {path = "../robin", develop = true}`. Adjust the path if your
> `robin` checkout lives elsewhere.

---

## Quick Start

### Run an experiment from the CLI

```bash
# Generate a supply (25 services, seed=42)
poetry run python scripts/run_supply_generator.py --n-services 25 --seed 42

# Optimize with the mealpy GA
poetry run python scripts/run_mealpy.py -a ga --seed 42 --epoch 50 --pop-size 20

# Optimize with the custom GSA
poetry run python scripts/run_gsa.py --seed 42 --epoch 50 --pop-size 20

# Optimize fairness-aware (revenue + Jain's index)
poetry run python scripts/run_fairness.py --alpha 0.7 --beta 0.3 --seed 42
```

Each optimization script:
1. Generates a supply (or loads one with `--supply path/to/supply.yaml`).
2. Builds the revenue-maximizing timetabling problem.
3. Runs the optimizer.
4. Saves the updated supply and convergence curve to
   `data/results/supply_updated_{algo}_seed{seed}.yaml` and
   `data/results/convergence_{algo}_seed{seed}.csv`.

### Run the notebooks

```bash
poetry run jupyter notebook notebooks/
```

| Notebook | Description |
|---|---|
| `1.0-Supply_Generator` | Generate a supply and visualize it with a Marey chart |
| `2.0-Capacity_Allocation` | Compare GA/DE/PSO/SCA across seeds; boxplots, convergence, Wilcoxon test |
| `3.0-Fairness_Timetabling` | Fairness-aware optimization (revenue + Jain/Gini/Atkinson) |
| `4.0-GSA_Timetabling` | Revenue maximization with the custom GSA |
| `5.0-Mealpy_Timetabling` | Revenue maximization with mealpy algorithms |

All notebooks are executable top-to-bottom.

---

## Architecture

```
src/craft/
  __init__.py        Public API + lazy backend imports
  common.py          Boundaries, Solution, ConflictMatrix (shared entities)
  revenue.py         RevenueSimulator, RevenueCalculator, penalty_function
  scheduling.py      ScheduleManager, get_schedule_from_supply
  fairness.py        FairnessMetrics (Jain / Gini / Atkinson)
  plotter.py         sns_box_plot, sns_line_plot, plot_scheduled_services
  utils.py           get_rus_revenue, is_better_solution
  runner.py          ExperimentConfig + ExperimentRunner (pipeline + CLI backend)
  gsa/
    algorithm.py     GSA optimizer
    elements.py      Velocity, Acceleration, GConstant
    fields.py        mass_calculation, gravitational constants, g_field
  mealpy/
    timetabling.py   MealpyTimetabling problem formulation
```

**Layering**: the shared layer (`common`, `revenue`, `scheduling`, `fairness`)
is independent; both optimization backends (`gsa`, `mealpy`) depend on it.
The `runner` orchestrates the full pipeline. See
[docs/architecture.md](docs/architecture.md) for details.

---

## Testing

```bash
poetry run pytest tests/ -v --cov=craft --cov-report=term-missing
```

82 tests covering all modules (entities, revenue, scheduling, fairness, GSA
fields/optimizer, runner config, and end-to-end smoke tests). Coverage: 70%.

---

## Project Layout

```
craft/
  configs/            Supply and generator configuration YAMLs
  data/results/       Generated supply YAMLs and convergence CSVs (gitignored)
  docs/               Project documentation
  notebooks/          Jupyter notebooks (1.0 → 5.0)
  reports/figures/    Generated plots (gitignored)
  scripts/            CLI entry points
  src/craft/          The craft package
  tests/              Pytest test suite
  pyproject.toml      Poetry project configuration
  poetry.lock         Locked dependencies
```

---

## Documentation

- [docs/architecture.md](docs/architecture.md) — architecture and data flow
- [docs/problem_formulation.md](docs/problem_formulation.md) — revenue and
  fairness formulation
- [docs/optimizers.md](docs/optimizers.md) — GSA vs mealpy backends and how to
  add new algorithms

---

## Authors

CRAFT is developed by David Muñoz Valero (`david.munoz6@alu.uclm.es`) as part
of the [MAT](https://blog.uclm.es/grupomat/) and
[ORETO](https://www.uclm.es/Home/Misiones/Investigacion/OfertaCientificoTecnica/GruposInvestigacion/DetalleGrupo?idgrupo=75)
research groups of the
[Escuela Superior de Informática](https://esi.uclm.es),
[University of Castilla-La Mancha (UCLM)](https://www.uclm.es).

## License

MIT
