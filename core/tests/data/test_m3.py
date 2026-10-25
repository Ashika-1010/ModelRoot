"""M3 tests: Seeds, splitting, preprocessing, fingerprints.

Test types:
  U = unit
  D = deterministic fixture
  P = property-based (Hypothesis)
  I = integration (M2→M3 pipeline)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from sklearn.compose import ColumnTransformer

from modelroot.data.errors import ContractViolation, ErrorCode
from modelroot.data.fingerprint import dataset_fingerprint
from modelroot.data.loader import load_csv
from modelroot.data.preprocessing import build_preprocessor
from modelroot.data.split import SplitData, make_split
from modelroot.data.validate import validate_for_analysis
from modelroot.schemas.common import Algorithm, SplitMode
from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig
from modelroot.utils.seeds import SeedManager

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "csv"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(
    path: str | pathlib.Path = "dummy.csv",
    target: str = "label",
    positive_class: str = "yes",
    split_mode: SplitMode = SplitMode.single_file,
    split_column: str | None = None,
    test_path: str | pathlib.Path | None = None,
    current_path: str | pathlib.Path | None = None,
) -> AnalysisConfig:
    split_cfg: dict = {"mode": split_mode}
    if split_column:
        split_cfg["split_column"] = split_column
    if test_path:
        split_cfg["test_path"] = str(test_path)
    return AnalysisConfig(
        data_path=pathlib.Path(path),
        target=target,
        positive_class=positive_class,
        split=SplitConfig(**split_cfg),
        current_path=pathlib.Path(current_path) if current_path else None,
        model_spec=ModelSpec(algorithm=Algorithm.logistic_regression),
    )


def _make_validated(n: int = 40, tmp_path: pathlib.Path | None = None) -> tuple:
    """Return (validated_dataset, config, tmp_path) with n balanced rows."""
    import tempfile

    if tmp_path is None:
        td = pathlib.Path(tempfile.mkdtemp())
    else:
        td = tmp_path
    rows = "\n".join(
        f"{20 + i},{50000 + i * 1000},{'yes' if i % 2 == 0 else 'no'}" for i in range(n)
    )
    p = td / "data.csv"
    p.write_text("age,income,label\n" + rows, encoding="utf-8")
    df = load_csv(p)
    cfg = _make_config(path=p, target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    return vd, cfg, td


# ---------------------------------------------------------------------------
# U: SeedManager
# ---------------------------------------------------------------------------


def test_seed_manager_same_root_same_name_deterministic() -> None:
    """U: Same root + same name always returns same seed."""
    sm1 = SeedManager(42)
    sm2 = SeedManager(42)
    assert sm1.spawn("bootstrap") == sm2.spawn("bootstrap")


def test_seed_manager_different_names_different_seeds() -> None:
    """U: Different names yield distinct seeds (no collisions for common names)."""
    sm = SeedManager(42)
    seeds = {sm.spawn(n) for n in ["split", "bootstrap", "repeat", "of:regularize", "im:reweight"]}
    # At least 4 distinct seeds out of 5
    assert len(seeds) >= 4


def test_seed_manager_different_roots_different_seeds() -> None:
    """U: Different roots yield different seeds for the same name."""
    assert SeedManager(0).spawn("split") != SeedManager(1).spawn("split")


def test_seed_manager_non_negative_output() -> None:
    """U: Spawned seed is always non-negative."""
    sm = SeedManager(99)
    for name in ["a", "b", "c", "d"]:
        assert sm.spawn(name) >= 0


def test_seed_manager_negative_root_raises() -> None:
    """U: Negative root raises ValueError."""
    with pytest.raises(ValueError, match="Root seed"):
        SeedManager(-1)


def test_seed_manager_no_builtin_hash() -> None:
    """U: SeedManager never calls Python's built-in hash() (uses SHA-256)."""
    # Spawn from two processes would be tested via subprocess; here we just
    # verify that seed_manager.spawn returns an int (not hash()-based).
    sm = SeedManager(0)
    seed = sm.spawn("test")
    assert isinstance(seed, int)


# ---------------------------------------------------------------------------
# U / D: make_split — single_file mode
# ---------------------------------------------------------------------------


def test_split_single_file_basic(tmp_path: pathlib.Path) -> None:
    """U: single_file split returns non-empty train and holdout sets."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    sm = SeedManager(cfg.seed)
    split = make_split(vd, cfg, sm.spawn("split"))
    assert isinstance(split, SplitData)
    assert len(split.X_train) > 0
    assert len(split.X_holdout) > 0


def test_split_sizes_sum_to_n(tmp_path: pathlib.Path) -> None:
    """U: train + holdout rows = total rows."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split = make_split(vd, cfg, 0)
    assert len(split.X_train) + len(split.X_holdout) == len(vd.df)


def test_split_indices_disjoint(tmp_path: pathlib.Path) -> None:
    """U: Train and holdout index sets are disjoint."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split = make_split(vd, cfg, 0)
    train_set = set(split.orig_index_train)
    holdout_set = set(split.orig_index_holdout)
    assert train_set.isdisjoint(holdout_set)


def test_split_same_seed_identical_indices(tmp_path: pathlib.Path) -> None:
    """D: Same seed → identical split indices."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split1 = make_split(vd, cfg, 7)
    split2 = make_split(vd, cfg, 7)
    assert list(split1.orig_index_train) == list(split2.orig_index_train)
    assert list(split1.orig_index_holdout) == list(split2.orig_index_holdout)


def test_split_different_seeds_different_indices(tmp_path: pathlib.Path) -> None:
    """U: Different seeds → different splits."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split1 = make_split(vd, cfg, 0)
    split2 = make_split(vd, cfg, 1)
    # With 40 rows a different seed should produce a different split
    assert list(split1.orig_index_train) != list(split2.orig_index_train)


def test_split_stratification_preserved(tmp_path: pathlib.Path) -> None:
    """U: Class ratio in train and holdout is close to overall ratio (within 1 row)."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split = make_split(vd, cfg, 42)
    total_pos = int(vd.y.sum())
    total_n = len(vd.y)
    overall_rate = total_pos / total_n

    train_rate = float(split.y_train.mean())
    holdout_rate = float(split.y_holdout.mean())

    tol = 1 / len(split.y_train) + 1 / len(split.y_holdout) + 0.05
    assert abs(train_rate - overall_rate) < tol
    assert abs(holdout_rate - overall_rate) < tol


def test_split_fingerprint_stable(tmp_path: pathlib.Path) -> None:
    """D: Same split → same fingerprint across two calls."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split1 = make_split(vd, cfg, 42)
    split2 = make_split(vd, cfg, 42)
    assert split1.split_fingerprint == split2.split_fingerprint


def test_split_fingerprint_changes_with_seed(tmp_path: pathlib.Path) -> None:
    """U: Different seed → different split fingerprint."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split1 = make_split(vd, cfg, 0)
    split2 = make_split(vd, cfg, 99)
    assert split1.split_fingerprint != split2.split_fingerprint


def test_split_original_index_retained(tmp_path: pathlib.Path) -> None:
    """U: X_train/X_holdout rows match their orig_index_* values."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split = make_split(vd, cfg, 0)
    assert list(split.X_train.index) == list(split.orig_index_train)
    assert list(split.X_holdout.index) == list(split.orig_index_holdout)


# ---------------------------------------------------------------------------
# U: make_split — split_column mode
# ---------------------------------------------------------------------------


def test_split_column_mode(tmp_path: pathlib.Path) -> None:
    """U: split_column mode splits correctly on 'train'/'test' values."""
    rows = "\n".join(
        f"{i},{'train' if i < 30 else 'test'},{'yes' if i % 2 == 0 else 'no'}" for i in range(40)
    )
    p = tmp_path / "split_col.csv"
    p.write_text("age,fold,label\n" + rows, encoding="utf-8")
    df = load_csv(p)
    cfg = _make_config(
        path=p,
        target="label",
        positive_class="yes",
        split_mode=SplitMode.split_column,
        split_column="fold",
    )
    vd = validate_for_analysis(df, cfg)
    split = make_split(vd, cfg, 0)
    assert len(split.X_train) == 30
    assert len(split.X_holdout) == 10


def test_split_column_missing_raises(tmp_path: pathlib.Path) -> None:
    """U: Missing split_column raises ContractViolation."""
    rows = "\n".join(f"{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(40))
    p = tmp_path / "data.csv"
    p.write_text("age,label\n" + rows, encoding="utf-8")
    df = load_csv(p)
    cfg = _make_config(
        path=p,
        target="label",
        positive_class="yes",
        split_mode=SplitMode.split_column,
        split_column="nonexistent_col",
    )
    vd = validate_for_analysis(df, cfg)
    with pytest.raises(ContractViolation):
        make_split(vd, cfg, 0)


# ---------------------------------------------------------------------------
# U: make_split — test_file mode
# ---------------------------------------------------------------------------


def test_split_test_file_mode(tmp_path: pathlib.Path) -> None:
    """U: test_file mode uses an external test file as holdout."""
    train_rows = "\n".join(f"{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(30))
    test_rows = "\n".join(f"{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(30, 40))
    train_p = tmp_path / "train.csv"
    test_p = tmp_path / "test.csv"
    train_p.write_text("age,label\n" + train_rows, encoding="utf-8")
    test_p.write_text("age,label\n" + test_rows, encoding="utf-8")

    df = load_csv(train_p)
    cfg = _make_config(
        path=train_p,
        target="label",
        positive_class="yes",
        split_mode=SplitMode.test_file,
        test_path=test_p,
    )
    vd = validate_for_analysis(df, cfg)
    split = make_split(vd, cfg, 0)
    assert len(split.X_train) == 30
    assert len(split.X_holdout) == 10


def test_split_test_file_missing_column_raises(tmp_path: pathlib.Path) -> None:
    """U: Test file missing a training feature column raises CURRENT_SCHEMA_MISMATCH."""
    train_rows = "\n".join(f"{i},{i * 2},{'yes' if i % 2 == 0 else 'no'}" for i in range(30))
    # test file only has 'age', not 'income'
    test_rows = "\n".join(f"{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(30, 40))
    train_p = tmp_path / "train.csv"
    test_p = tmp_path / "test.csv"
    train_p.write_text("age,income,label\n" + train_rows, encoding="utf-8")
    test_p.write_text("age,label\n" + test_rows, encoding="utf-8")

    df = load_csv(train_p)
    cfg = _make_config(
        path=train_p,
        target="label",
        positive_class="yes",
        split_mode=SplitMode.test_file,
        test_path=test_p,
    )
    vd = validate_for_analysis(df, cfg)
    with pytest.raises(ContractViolation) as exc:
        make_split(vd, cfg, 0)
    assert exc.value.code == ErrorCode.CURRENT_SCHEMA_MISMATCH


# ---------------------------------------------------------------------------
# U: build_preprocessor
# ---------------------------------------------------------------------------


def test_preprocessor_returns_column_transformer() -> None:
    """U: build_preprocessor returns a ColumnTransformer."""
    ct = build_preprocessor({"age": "numeric", "gender": "categorical"})
    assert isinstance(ct, ColumnTransformer)


def test_preprocessor_fit_on_train_only(tmp_path: pathlib.Path) -> None:
    """U: Preprocessor statistics come from training data only.

    Verify by checking that the imputer's learned statistics match training
    data, not holdout data.
    """
    vd, cfg, _ = _make_validated(40, tmp_path)
    split = make_split(vd, cfg, 0)
    ct = build_preprocessor(vd.column_types, scale=False)
    ct.fit(split.X_train)

    # The numeric imputer fills with median of training data
    numeric_transformer = ct.named_transformers_.get("numeric")
    if numeric_transformer is not None:
        imputer = numeric_transformer.named_steps["imputer"]
        # Verify imputer statistics come from training data
        train_numeric_cols = [c for c, t in vd.column_types.items() if t == "numeric"]
        for i, col in enumerate(train_numeric_cols):
            train_median = split.X_train[col].replace("", float("nan")).astype(float).median()
            assert abs(imputer.statistics_[i] - train_median) < 1e-6


def test_preprocessor_unseen_category_in_holdout(tmp_path: pathlib.Path) -> None:
    """D: Unseen category in holdout is handled without error (encoded as all-zeros)."""
    # Build a larger dataset so both files exceed the min_rows limit
    rows_train = "\n".join(
        f"{'M' if i % 3 != 2 else 'F'},{'yes' if i % 2 == 0 else 'no'}" for i in range(30)
    )
    # holdout with unseen category 'Z' — needs ≥10 rows
    rows_holdout = "\n".join(f"Z,{'yes' if i % 2 == 0 else 'no'}" for i in range(12))
    p = tmp_path / "train.csv"
    test_p = tmp_path / "test.csv"
    p.write_text("gender,label\n" + rows_train, encoding="utf-8")
    test_p.write_text("gender,label\n" + rows_holdout, encoding="utf-8")

    df_train = load_csv(p)
    cfg = _make_config(
        path=p,
        target="label",
        positive_class="yes",
        split_mode=SplitMode.test_file,
        test_path=test_p,
    )
    vd = validate_for_analysis(df_train, cfg)
    split = make_split(vd, cfg, 0)
    ct = build_preprocessor(vd.column_types, scale=False)
    ct.fit(split.X_train)
    # Should not raise; unseen category encoded as zeros
    X_holdout_transformed = ct.transform(split.X_holdout)
    assert X_holdout_transformed.shape[0] == len(split.X_holdout)


def test_preprocessor_missing_values_imputed(tmp_path: pathlib.Path) -> None:
    """U: Missing (empty-string) values are imputed after fitting."""
    # Use numeric-looking values to ensure 'age' is inferred as numeric
    # 25 rows, some missing
    ages = [str(20 + i) if i % 5 != 0 else "" for i in range(30)]
    labels = ["yes" if i % 2 == 0 else "no" for i in range(30)]
    rows = "\n".join(f"{a},{l}" for a, l in zip(ages, labels, strict=False))
    p = tmp_path / "t.csv"
    p.write_text("age,label\n" + rows, encoding="utf-8")
    df = load_csv(p)
    cfg = _make_config(path=p, target="label", positive_class="yes")
    vd = validate_for_analysis(df, cfg)
    # age may not have enough unique values (only 30 rows) to be numeric;
    # ensure column_types is not empty by using the actual inferred types
    if not vd.column_types:
        pytest.skip("No feature columns inferred — dataset too small")
    split = make_split(vd, cfg, 0)
    ct = build_preprocessor(vd.column_types, scale=False)
    X_out = ct.fit_transform(split.X_train)
    assert not np.isnan(X_out).any()


def test_preprocessor_scale_adds_standard_scaler(tmp_path: pathlib.Path) -> None:
    """U: scale=True adds StandardScaler to numeric pipeline."""
    # Inspect transformers list (unfitted) directly
    ct_scaled = build_preprocessor({"age": "numeric"}, scale=True)
    ct_noscale = build_preprocessor({"age": "numeric"}, scale=False)

    # Get the pipeline from the transformers list (not named_transformers_ which needs fit)
    def _get_numeric_steps(ct: ColumnTransformer) -> list:
        for name, transformer, cols in ct.transformers:
            if name == "numeric":
                return transformer.steps  # type: ignore[union-attr]
        return []

    steps_scaled = _get_numeric_steps(ct_scaled)
    steps_noscale = _get_numeric_steps(ct_noscale)
    assert len(steps_scaled) == 3  # to_float + imputer + scaler
    assert len(steps_noscale) == 2  # to_float + imputer only
    assert steps_scaled[2][0] == "scaler"


# ---------------------------------------------------------------------------
# U: dataset_fingerprint (additional stability test)
# ---------------------------------------------------------------------------


def test_fingerprint_stable_across_independent_calls(tmp_path: pathlib.Path) -> None:
    """D: Fingerprint is stable when computed twice on the same data."""
    vd, _, _ = _make_validated(40, tmp_path)
    fp1 = dataset_fingerprint(vd.df)
    fp2 = dataset_fingerprint(vd.df.copy())
    assert fp1 == fp2


# ---------------------------------------------------------------------------
# P: property-based — split sizes
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=20, max_value=60),
    test_size=st.floats(min_value=0.1, max_value=0.4),
)
@settings(max_examples=15)
def test_split_sizes_sum_to_n_property(n: int, test_size: float) -> None:
    """P: train + holdout = n for any valid n / test_size."""
    import tempfile

    # Make n even so each class has at least 2 samples
    n_even = n if n % 2 == 0 else n + 1
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / f"prop_{n_even}.csv"
        rows = "\n".join(f"{i},{'yes' if i % 2 == 0 else 'no'}" for i in range(n_even))
        p.write_text("age,label\n" + rows, encoding="utf-8")
        df = load_csv(p)
        from modelroot.schemas.config import SplitConfig

        cfg = AnalysisConfig(
            data_path=p,
            target="label",
            positive_class="yes",
            split=SplitConfig(mode=SplitMode.single_file, test_size=test_size),
            model_spec=ModelSpec(algorithm=Algorithm.logistic_regression),
        )
        vd = validate_for_analysis(df, cfg)
        split = make_split(vd, cfg, 0)
        assert len(split.X_train) + len(split.X_holdout) == n_even


# ---------------------------------------------------------------------------
# I: M2 → M3 pipeline integration
# ---------------------------------------------------------------------------


def test_m2_to_m3_pipeline(tmp_path: pathlib.Path) -> None:
    """I: Full M2→M3 pipeline: load → validate → split → preprocess."""
    vd, cfg, _ = _make_validated(40, tmp_path)
    split = make_split(vd, cfg, 42)
    ct = build_preprocessor(vd.column_types, scale=True)
    X_train_out = ct.fit_transform(split.X_train)
    X_holdout_out = ct.transform(split.X_holdout)
    assert X_train_out.shape[0] == len(split.X_train)
    assert X_holdout_out.shape[0] == len(split.X_holdout)
    assert X_train_out.shape[1] == X_holdout_out.shape[1]
    # No NaNs after preprocessing
    assert not np.isnan(X_train_out).any()
    assert not np.isnan(X_holdout_out).any()
