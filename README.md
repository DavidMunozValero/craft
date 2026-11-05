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

CRAFT depends on the [`robin`](https://github.com/JoseAngelMartinB/robin)
railway simulator, which is **not consumed from PyPI**: it is installed as an
editable path dependency from a local clone of its `master` branch. This is the
rough edge of the setup — robin's own installation instructions are outdated
(the conda `environment.yml` referenced in its README does not exist) — so the
steps below cover it explicitly.

### Prerequisites

- Python ≥ 3.11 (3.11 recommended)
- [Poetry](https://python-poetry.org/) ≥ 2.0 (or `pip`, see the fallback below)
- `git`

### 1. Clone `robin` — required before installing CRAFT

The path dependency `../robin` is resolved at install time, so the clone must
exist **before** running `poetry install`:

```bash
cd /path/to/your/projects          # parent dir of the craft checkout
git clone https://github.com/JoseAngelMartinB/robin.git
```

Expected layout:

```
projects/
├── craft/    ← this repository
└── robin/    ← simulator checkout (default branch: master)
```

If your `robin` clone lives elsewhere, edit the path in `pyproject.toml`
(`robin = {path = "../robin", develop = true}`) and run `poetry lock` after
changing it.

### 2. Create the virtual environment and install

```bash
cd /path/to/your/projects/craft
poetry lock       # resolve craft + robin into poetry.lock
poetry install    # create the virtualenv and install everything
```

This creates a single Poetry-managed virtualenv with:

- `robin` in **editable mode** from `../robin` (a `.pth` file pointing at its
  `src/`),
- the runtime dependencies of both projects — note that robin drags in heavy
  extras such as `selenium`, `optuna`, `shapely` and `networkx` (~150 packages),
- the dev tools (`pytest`, `pytest-cov`, `ruff`, `mypy`, `jupyter`),
- CRAFT itself, also in editable mode.

### 3. Verify the installation

```bash
poetry run python -c "import craft, robin; print(robin.__file__)"
```

`robin.__file__` must point inside `../robin/src/robin/`. If it points into
`site-packages/robin/`, the editable install did not apply (e.g. a PyPI copy of
robin was picked up first) — recreate the virtualenv (`poetry env remove && poetry install`).

Then run the test suite:

```bash
poetry run pytest tests/ -v
```

### The `robin` dependency: what you should know

- **Editable, not pinned.** The lock entry is `develop = true` with no files:
  Poetry installs the checkout itself, not a frozen copy. Code changes in
  `../robin` take effect on the next interpreter start — no reinstall, no
  re-lock needed.
- **It is not updated by `poetry update`.** To track upstream:
  ```bash
  git -C ../robin pull
  poetry run pytest tests/ -v    # CRAFT is only guaranteed against robin master
  ```
  If robin's *own* `pyproject.toml` dependencies changed, also run
  `poetry lock && poetry install` so the lock picks up the new constraints.
- **Do not `pip install robin` from PyPI.** The simulator must come from the
  git checkout; a stray PyPI copy shadows the editable one.
- The local checkout may be on any branch, but `master` is what CRAFT is
  tested against.

### Manual installation without Poetry (fallback)

For machines where Poetry is not available. From the parent directory that
contains both `craft/` and `robin/`:

```bash
python3.11 -m venv craft/.venv
source craft/.venv/bin/activate
pip install -e robin    # robin + its dependencies first
pip install -e craft    # then CRAFT (mealpy, numpy, pandas, ...)
pip install pytest pytest-cov ruff mypy jupyter   # dev tools
```

Install `robin` first so `pip` reuses its already-resolved scientific stack
instead of backtracking. When you work outside Poetry, activate the venv with
`source craft/.venv/bin/activate`.

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
| `0.0-Synthetic_Data_Generator` | Synthetic timetable + canon/penalties for the Madrid–Barcelona line (two standalone YAMLs in `data/synthetic/`) |
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
  data/synthetic/     Synthetic timetable/revenue YAMLs for the Madrid–Barcelona line (gitignored)
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

### Empezando desde cero (papers ↔ código)

1. [docs/research_context.md](docs/research_context.md) — contexto de
   investigación: problema, los dos papers del proyecto y sus formulaciones.
2. [docs/papers/README.md](docs/papers/README.md) — guía de las fuentes
   LaTeX de los papers (versiones y cuál leer).
3. [docs/paper_code_mapping.md](docs/paper_code_mapping.md) — mapeo
   ecuación a ecuación entre papers y módulos, con discrepancias.
4. [docs/audit_2026-11.md](docs/audit_2026-11.md) — auditoría del
   repositorio y backlog priorizado.

### Referencia técnica

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
