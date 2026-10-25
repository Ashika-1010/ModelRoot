# Changelog

All notable changes to ModelRoot will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versions correspond to milestones; semantic versioning begins at v1.0 (M36).

---

## [Unreleased — M1 audit fixes] — 2026-10-10

### Changed

- `AnalysisResult.experiments` field type corrected from `list[ExperimentSpec]` to
  `list[ExperimentResult]`.  The master plan §3.3 specifies that the pipeline runner
  returns `ExperimentResult` objects; `AnalysisResult` is the complete output artifact
  and must carry the full experiment outcomes, not just the input specifications.
- `AblationConfig` removed from the public export registry (`export_json_schemas`)
  and from `modelroot.schemas.__all__`.  It remains accessible as
  `modelroot.schemas.config.AblationConfig` and is reachable via `$defs` in the
  exported `AnalysisConfig` JSON Schema.  It was never a standalone top-level
  contract in the master plan; it was an inline sub-object of `AnalysisConfig`.
- `docs/schemas/examples/analysis_result.json` updated so `experiments[]` contains
  an `ExperimentResult` object consistent with the corrected field type.
- JSON Schema snapshots regenerated: `AnalysisResult.json` updated to reflect
  `ExperimentResult` elements; `AblationConfig.json` snapshot removed.




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

## [Unreleased — M2] — 2026-10-10

### Added

- `core/src/modelroot/data/` package: `errors.py`, `loader.py`, `validate.py`,
  `profile.py`, `fingerprint.py`.
- `Limits` dataclass (50 MB file, 50K rows, 50 cols, min 10 rows).
- `load_csv(path, limits)` with file-size, null-byte, encoding (UTF-8/UTF-8-BOM),
  duplicate-header, row-count, and column-count guards.
- `validate_for_analysis(df, config)` returning `ValidatedDataset` with target
  extraction (y in {0,1}), class-size validation, column type inference (numeric
  if >=95% parse + >10 unique, else categorical), datetime exclusion, constant
  column dropping, high-missingness warnings, and id_column exclusion (F5).
- `profile(df, column_types, target_col)` returning `DatasetProfile`.
- `ContractViolation(code, message)` with 13 error codes.
- `dataset_fingerprint(df)` returning stable SHA-256 hex.
- Runtime deps: `pandas>=2.0,<3.0`, `numpy>=1.26,<3.0`.
- Tests: 45 tests covering all error codes, type inference, security, profiling,
  fingerprints, and property-based column-order invariance.

## [Unreleased — M3] — 2026-10-10

### Added

- `core/src/modelroot/utils/seeds.py`: `SeedManager(root).spawn(name)` using
  SHA-256 name hash + SeedSequence (no Python `hash()`).
- `core/src/modelroot/data/split.py`: `make_split()` supporting `single_file`
  (stratified), `split_column`, and `test_file` modes; optional `current`
  dataset alignment with `CURRENT_SCHEMA_MISMATCH` error code; original row
  index retention; split fingerprint.
- `core/src/modelroot/data/preprocessing.py`: `build_preprocessor(column_types, scale)`
  returning unfitted `ColumnTransformer` with median imputation + optional
  `StandardScaler` for numeric, most-frequent imputation + `OneHotEncoder`
  (handle_unknown="ignore", min_frequency=0.01, max_categories=50) for
  categorical. Includes `_StringToFloat` transformer for raw-string DataFrames.
- `SplitData` dataclass with `X_train`, `y_train`, `X_holdout`, `y_holdout`,
  `X_current?`, `y_current?`, `orig_index_*`, `split_fingerprint`, `warnings`.
- Runtime dep: `scikit-learn>=1.4,<2.0`.
- Tests: 27 tests covering seed determinism, all split modes, stratification,
  disjoint indices, fingerprint stability, unseen-category handling,
  train-only preprocessing, missing-value imputation, and property-based
  split-size invariants.

## [Unreleased — M4] — 2026-10-10

### Added

- `core/src/modelroot/models/` package: `adapter.py` (abstract `ModelAdapter`
  protocol), `registry.py` (`get_model`, `list_model_specs`, `InvalidModelSpec`),
  `sklearn_adapters.py` (4 concrete adapters).
- Four adapters: `LogisticRegressionAdapter` (with StandardScaler),
  `DecisionTreeAdapter`, `RandomForestAdapter`, `HistGradientBoostingAdapter`
  (no scaling for trees).
- Each adapter wraps preprocessor + estimator in sklearn `Pipeline`; validates
  params against allow-list (types, ranges, values); rejects injection strings;
  `random_state` always set from seed.
- `clone_with(**overrides)` returns new unfitted adapter (original unchanged).
- `capacity_grid()` per §5-H1: LR C in {1,.1,.01,.001}; DT max_depth x
  min_samples_leaf; RF min_samples_leaf x max_depth x max_features; HGB
  max_depth x l2 x learning_rate.
- `supports_sample_weight()`, `complexity()` per adapter.
- Tests: 46 tests covering fit/predict/proba, proba shape & sum=1, same-seed
  determinism, clone immutability, capacity grids, param validation, security
  rejection, sample weights, and full M2->M3->M4 pipeline for all 4 algorithms.

### Changed

- `mypy` dev dependency updated from `==1.15.0` to `==1.18.2` to resolve
  numpy 2.x bundled stub incompatibility (PEP 695 `type` statement syntax
  requires Python 3.12+ for parsing, incompatible with mypy 1.15 +
  `python_version="3.11"`).
- `pyproject.toml` `[tool.mypy]` `python_version` set to `"3.12"` (tooling
  only; runtime remains `>=3.11`). Our source code uses no 3.12-only syntax.
- `pytest.ini` `filterwarnings` added to suppress pandas datetime UserWarning.
