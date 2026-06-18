# AGENTS.md

Guidance for AI agents (and humans) working on this repository.

## Commands

### Install
```bash
poetry lock && poetry install
```

### Test
```bash
poetry run pytest tests/ -v
poetry run pytest tests/ -v --cov=craft --cov-report=term-missing   # with coverage
poetry run pytest tests/test_gsa.py -v                                # single module
```

### Lint
```bash
poetry run ruff check src/ tests/ scripts/
poetry run ruff check --fix src/ tests/ scripts/   # auto-fix
```

### Format
```bash
poetry run ruff format src/ tests/ scripts/
```

### Type check
```bash
poetry run mypy src/craft/
```

### Run scripts
```bash
poetry run python scripts/run_supply_generator.py --n-services 25 --seed 42
poetry run python scripts/run_mealpy.py -a ga --seed 42 --epoch 50 --pop-size 20
poetry run python scripts/run_gsa.py --seed 42 --epoch 50 --pop-size 20
poetry run python scripts/run_fairness.py --alpha 0.7 --beta 0.3 --seed 42
```

### Execute notebooks (validation)
```bash
poetry run jupyter nbconvert --to notebook --execute --output /tmp/check.ipynb notebooks/5.0-Mealpy_Timetabling.ipynb
```

## Architecture

- `src/craft/` — the package (src layout). Shared layer: `common`, `revenue`,
  `scheduling`, `fairness`. Backends: `gsa/`, `mealpy/`. Orchestrator:
  `runner.py`.
- `robin` is an editable path dependency from `../robin` (GitHub master).
- Backends (`craft.GSA`, `craft.MealpyTimetabling`) are lazy-loaded via
  `__getattr__` in `__init__.py` to keep `import craft` light.

## Conventions

- No comments in code unless explicitly requested.
- Imports inside the package are relative (`from ..common import ...`).
- Imports in notebooks/scripts are absolute (`from craft import ...`).
- mealpy 3.x API: `epoch`/`pop_size` in the constructor, `seed` in `solve()`.
  Correct class names: `DE.OriginalDE`, `SCA.OriginalSCA` (not `DE.Original`).
- GSA objective functions must return a `(fitness, accuracy)` tuple.
- mealpy objective functions must return a scalar `fitness`.
- GSA optimizes only real variables; discrete (scheduled mask) comes from
  `timetabling.get_heuristic_schedule()`.
- `pop_size >= 10` is required for mealpy GA (k-way tournament selection).

## Before committing

1. `poetry run ruff check src/ tests/ scripts/`
2. `poetry run pytest tests/ -v`
3. Do not commit `data/results/` or `reports/figures/` (gitignored).
