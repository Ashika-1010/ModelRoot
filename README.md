# ModelRoot

Reproducible diagnosis engine for underperforming binary classifiers.

ModelRoot takes a binary-classification CSV and a model specification, trains
the baseline itself, and runs targeted experiments to assess whether observed
underperformance is explained by overfitting, class imbalance, data leakage,
subgroup gaps, or distribution shift.

> **Status:** Milestone M0 complete — repository skeleton, tooling, and CI.
> No diagnosis functionality exists yet.

---

## Repository layout

```
ModelRoot/
├── core/                   # Python package (modelroot)
│   ├── pyproject.toml
│   └── src/modelroot/
├── docs/
│   └── MASTER_PLAN.md      # Full implementation plan (M0–M36)
└── .github/workflows/      # GitHub Actions CI
```

---

## Setup

Requires **Python 3.11+**.  All commands below are run from the repository
root.

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows PowerShell

# 2. Install the package with development dependencies
pip install -e "core[dev]"

# 3. (Optional) Install pre-commit hooks
pre-commit install
```

---

## Development checks

```bash
# Full fast-CI gate: lint + type-check + tests
make check

# Individual targets
make lint        # ruff check + format check
make type        # mypy strict on core/
make test        # pytest (excludes slow / docker / e2e)
make test-slow   # pytest including slow tests
```

Run `pytest --markers` to list all registered custom markers.

---

## Running tests directly

```bash
# From the repository root (package must be installed first)
pytest core/tests/

# With coverage
pytest --cov=modelroot --cov-report=term-missing core/tests/
```
