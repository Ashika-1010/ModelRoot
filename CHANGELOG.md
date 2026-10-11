# Changelog

All notable changes to ModelRoot will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to milestones; semantic versioning begins at v1.0 (M36).

---

## [Unreleased — M0] — 2026-10-10

### Added

- Repository skeleton: `README.md`, `LICENSE` (placeholder), `CHANGELOG.md`,
  `.gitignore`, `Makefile`.
- `core/` Python package (`modelroot`) with `src` layout, installable via
  `pip install -e "core[dev]"`.
- `core/src/modelroot/__init__.py` exposing `__version__ = "0.0.1"`.
- `core/pyproject.toml` with Python ≥3.11 requirement and development
  dependencies: `pytest`, `pytest-cov`, `hypothesis`, `ruff`, `mypy`.
- Pytest configured with `--strict-markers` and six custom markers:
  `slow`, `integration`, `bench`, `security`, `docker`, `e2e`.
- `core/tests/test_smoke.py`: import/version test and strict-marker
  enforcement test (subprocess-based behavioural check).
- `Makefile` targets: `check`, `lint`, `type`, `test`, `test-slow`.
- `.pre-commit-config.yaml` with `pre-commit-hooks`, `ruff`, and `mypy`
  hooks pinned to the same versions as the project dev dependencies.
- `.github/workflows/ci.yml`: CI pipeline running `make check` on every push
  and pull request using Python 3.11 with pip caching.
- `docs/MASTER_PLAN.md`: full ModelRoot implementation master plan (M0–M36).

### Changed

- Nothing (initial milestone).

### Removed

- Nothing (initial milestone).
