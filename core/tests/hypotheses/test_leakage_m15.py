"""Tests for M15 Leakage hypotheses: Duplicate (H3a) and Identifier (H3b)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pandas as pd

from modelroot.assessment.baseline import BaselineRun, PredictionSet
from modelroot.config.thresholds import Thresholds
from modelroot.hypotheses.base import HypothesisContext
from modelroot.hypotheses.leakage import (
    LeakageDuplicateHypothesis,
    LeakageIdentifierHypothesis,
    _is_continuous_numeric,
    permutation_importance_share,
    row_hashes,
)
from modelroot.schemas.common import PrimaryMetric, RulesProfile, SplitMode, Status
from modelroot.utils.seeds import SeedManager


def _make_dummy_ctx(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_ho: pd.DataFrame,
    y_ho: pd.Series,
    fitted_model: Any,
    orig_idx_tr: pd.Index | None = None,
    split_mode: SplitMode = SplitMode.single_file,
    metric: PrimaryMetric = PrimaryMetric.accuracy,
    budget_check: Any = None,
) -> HypothesisContext:
    split = MagicMock()
    split.X_train = X_tr
    split.y_train = y_tr
    split.X_holdout = X_ho
    split.y_holdout = y_ho
    split.orig_index_train = orig_idx_tr if orig_idx_tr is not None else X_tr.index
    split.orig_index_holdout = X_ho.index

    y_tr_pred = (
        fitted_model.predict(X_tr)
        if hasattr(fitted_model, "predict")
        else np.zeros(len(y_tr), dtype=int)
    )
    y_tr_proba = (
        fitted_model.predict_proba(X_tr)[:, 1]
        if hasattr(fitted_model, "predict_proba") and not isinstance(fitted_model, MagicMock)
        else np.zeros(len(y_tr), dtype=float)
    )
    y_ho_pred = (
        fitted_model.predict(X_ho)
        if hasattr(fitted_model, "predict")
        else np.zeros(len(y_ho), dtype=int)
    )
    y_ho_proba = (
        fitted_model.predict_proba(X_ho)[:, 1]
        if hasattr(fitted_model, "predict_proba") and not isinstance(fitted_model, MagicMock)
        else np.zeros(len(y_ho), dtype=float)
    )
    train_pred = PredictionSet(
        index=X_tr.index,
        y_true=y_tr.to_numpy(),
        y_pred=y_tr_pred,
        y_proba=y_tr_proba,
    )
    holdout_pred = PredictionSet(
        index=X_ho.index,
        y_true=y_ho.to_numpy(),
        y_pred=y_ho_pred,
        y_proba=y_ho_proba,
    )
    baseline_run = BaselineRun(
        adapter=fitted_model,
        train_pred=train_pred,
        holdout_pred=holdout_pred,
        current_pred=None,
        metrics=[],
    )

    cfg = MagicMock()
    cfg.primary_metric = metric
    cfg.rules_profile = RulesProfile.full
    cfg.split = MagicMock()
    cfg.split.mode = split_mode
    cfg.model_spec = {"algorithm": "decision_tree", "params": {"max_depth": 3}}
    cfg.seed = 42

    col_types = dict.fromkeys(X_tr.columns, "numeric")
    sm = SeedManager(42)

    return HypothesisContext(
        split=split,
        baseline_run=baseline_run,
        config=cfg,
        col_types=col_types,
        seeds=sm,
        thresholds=Thresholds(),
        budget_check=budget_check,
    )


# ---------------------------------------------------------------------------
# row_hashes tests
# ---------------------------------------------------------------------------


def test_row_hashes_properties() -> None:
    """Validate uint64 dtype, numeric normalization, NaN preservation, immutability."""
    # Numerics: float vs int should produce identical hashes
    df1 = pd.DataFrame({"a": [1, 2, 3], "b": [10.0, np.nan, 30.0]})
    df2 = pd.DataFrame({"a": [1.0, 2.0, 3.0], "b": [10, np.nan, 30]})

    h1 = row_hashes(df1)
    h2 = row_hashes(df2)

    assert h1.dtype == np.uint64
    assert np.array_equal(h1, h2)

    # Immutability: df1 column dtypes unchanged
    assert pd.api.types.is_integer_dtype(df1["a"])

    # Duplicate detection: row 0 duplicate of row 3
    df3 = pd.DataFrame({"x": ["cat", "dog", "bird", "cat"], "y": [1.0, 2.0, 3.0, 1.0]})
    h3 = row_hashes(df3)
    assert h3[0] == h3[3]
    assert h3[0] != h3[1]


# ---------------------------------------------------------------------------
# H3a Duplicate Leakage tests
# ---------------------------------------------------------------------------


def test_leakage_duplicate_clean_dataset() -> None:
    """Clean dataset has 0 duplicate rows -> not_triggered."""
    X_tr = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0], "f2": [10.0, 20.0, 30.0, 40.0]})
    y_tr = pd.Series([0, 1, 0, 1])
    X_ho = pd.DataFrame({"f1": [5.0, 6.0, 7.0, 8.0], "f2": [50.0, 60.0, 70.0, 80.0]})
    y_ho = pd.Series([0, 1, 0, 1])

    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0, 1])

    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model)
    hyp = LeakageDuplicateHypothesis()

    trig = hyp.trigger(ctx)
    assert not trig.fired


def test_is_continuous_numeric_unit() -> None:
    """Unit tests for _is_continuous_numeric logic."""
    # 1. Native floats
    s_float = pd.Series([1.23, 4.56, 7.89, -0.45])
    assert _is_continuous_numeric(s_float) is True
    s_float32 = pd.Series([1.23, 4.56, 7.89], dtype=np.float32)
    assert _is_continuous_numeric(s_float32) is True

    # Native floats that are actually whole integers
    s_float_integers = pd.Series([1.0, 2.0, 3.0, 4.0])
    assert _is_continuous_numeric(s_float_integers) is False

    # 2. String-encoded continuous numeric columns
    s_str_float = pd.Series(["-0.123", "1.456", "3.1415", "99.01", "0.005"])
    assert _is_continuous_numeric(s_str_float) is True

    # 3. Genuine identifier columns (integers, string integers, alphanumeric tokens)
    s_int = pd.Series(np.arange(100), dtype=int)
    assert _is_continuous_numeric(s_int) is False

    s_str_int = pd.Series([str(i) for i in range(100)])
    assert _is_continuous_numeric(s_str_int) is False

    s_tokens = pd.Series([f"USR_{i:04d}" for i in range(100)])
    assert _is_continuous_numeric(s_tokens) is False

    # 4. Missing and mixed values
    s_mixed = pd.Series(["1.23", None, "", "4.56", "NaN", "null", "7.89", "N/A"])
    assert _is_continuous_numeric(s_mixed) is True

    # Mostly non-numeric string column
    s_mostly_str = pd.Series(["apple", "banana", "1.23", "cherry", "date"])
    assert _is_continuous_numeric(s_mostly_str) is False

    # Integer series with NaN (pandas casts integer with NaN to float64)
    s_int_with_nan = pd.Series([1.0, 2.0, np.nan, 4.0, 5.0])
    assert _is_continuous_numeric(s_int_with_nan) is False


def test_leakage_identifier_trigger_skips_string_continuous_features() -> None:
    """String-encoded continuous features with high cardinality should NOT trigger S1."""
    n = 200
    rng = np.random.default_rng(42)
    cont_floats = rng.standard_normal(n)
    str_floats = [f"{v:.6f}" for v in cont_floats]
    native_floats = rng.standard_normal(n)

    X = pd.DataFrame(
        {
            "feat_cont_str": str_floats,
            "feat_cont_num": native_floats,
        }
    )
    y = rng.integers(0, 2, size=n)

    model = MagicMock()
    model.predict.return_value = y

    ctx = _make_dummy_ctx(X, pd.Series(y), X, pd.Series(y), model)
    hyp = LeakageIdentifierHypothesis()

    trig = hyp.trigger(ctx)
    assert not trig.fired, f"Trigger fired unexpectedly: {trig.values}"


def test_leakage_identifier_trigger_detects_genuine_string_and_int_identifiers() -> None:
    """Genuine string tokens, integer IDs, and string-encoded integer IDs trigger S1."""
    n = 200
    int_ids = np.arange(n)
    str_int_ids = [str(i) for i in range(n)]
    token_ids = [f"ID_{i:05d}" for i in range(n)]

    rng = np.random.default_rng(42)
    cont_feat = [f"{v:.5f}" for v in rng.standard_normal(n)]

    X = pd.DataFrame(
        {
            "id_int": int_ids,
            "id_str_int": str_int_ids,
            "id_token": token_ids,
            "feat_cont": cont_feat,
        }
    )
    y = rng.integers(0, 2, size=n)

    model = MagicMock()
    model.predict.return_value = y

    ctx = _make_dummy_ctx(X, pd.Series(y), X, pd.Series(y), model)
    hyp = LeakageIdentifierHypothesis()

    trig = hyp.trigger(ctx)
    assert trig.fired
    # flagged_features is stored as a comma-separated string
    flagged_str = trig.values.get("flagged_features") or ""
    flagged = {f for f in str(flagged_str).split(",") if f}
    assert "id_int" in flagged
    assert "id_str_int" in flagged
    assert "id_token" in flagged
    assert "feat_cont" not in flagged

    # Evaluate should produce an identifier result
    res = hyp.evaluate(ctx, trig, [])
    assert len(res) == 1
    assert res[0].subtype == "identifier"
    # Status is determined by importance share; with MagicMock model it may be any non-fired value
    assert res[0].status != Status.not_triggered


def test_leakage_duplicate_detected() -> None:
    """Holdout contains duplicated training rows inflating accuracy."""
    rng = np.random.default_rng(42)
    # 200 train rows
    X_tr = pd.DataFrame(rng.normal(size=(200, 4)), columns=[f"f{i}" for i in range(4)])
    y_tr = pd.Series(rng.integers(0, 2, size=200))

    # Holdout: 250 clean rows (to meet power gate n >= 200), plus 50 duplicated rows from train
    X_ho_clean = pd.DataFrame(rng.normal(size=(250, 4)), columns=[f"f{i}" for i in range(4)])
    y_ho_clean = pd.Series(rng.integers(0, 2, size=250))

    X_ho_dup = X_tr.iloc[:50].copy()
    y_ho_dup = y_tr.iloc[:50].copy()

    X_ho = pd.concat([X_ho_clean, X_ho_dup], ignore_index=True)
    y_ho = pd.concat([y_ho_clean, y_ho_dup], ignore_index=True)

    # Model perfectly predicts the duplicate training rows (1.0 acc), but 0.5 on clean
    model = MagicMock()
    pred_clean = np.array([1, 0] * 125)  # ~50%
    pred_dup = y_ho_dup.to_numpy()  # 100%
    full_pred = np.concatenate([pred_clean, pred_dup])
    model.predict.return_value = full_pred

    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model)
    hyp = LeakageDuplicateHypothesis()

    trig = hyp.trigger(ctx)
    assert trig.fired
    # 50 dups out of 150 = 33.3% >= 1%
    assert trig.values["dup_fraction"] > 0.10

    res = hyp.evaluate(ctx, trig, [])
    assert len(res) == 1
    hr = res[0]
    assert hr.subtype == "duplicate"
    # Inflation delta = full (66.6%) - deduplicated (50%) = ~16.6% >= 2% delta_min
    assert hr.effect is not None
    assert hr.effect.delta > 0.05
    assert hr.status in (Status.supported, Status.weakly_supported)
    assert "Exact duplicates only" in hr.caveats[0]
    assert "Remove duplicate rows" in hr.recommended_actions[0]


def test_leakage_duplicate_underpowered_subset() -> None:
    """If deduplicated subset has < 2 rows, evaluation returns inconclusive."""
    X_tr = pd.DataFrame({"f": [1.0, 2.0, 3.0]})
    y_tr = pd.Series([0, 1, 0])
    # Holdout is almost entirely duplicated, leaving only 1 non-duplicate
    X_ho = pd.DataFrame({"f": [1.0, 2.0, 99.0]})
    y_ho = pd.Series([0, 1, 0])

    model = MagicMock()
    model.predict.return_value = np.array([0, 1, 0])

    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model)
    hyp = LeakageDuplicateHypothesis()
    trig = hyp.trigger(ctx)
    assert trig.fired

    res = hyp.evaluate(ctx, trig, [])
    assert res[0].status == Status.inconclusive


# ---------------------------------------------------------------------------
# H3b Identifier Leakage tests
# ---------------------------------------------------------------------------


def test_permutation_importance_share_math() -> None:
    """Validate importance share normalization and no mutation."""
    from sklearn.tree import DecisionTreeClassifier

    # f0 is perfectly predictive, f1 is noise
    X = pd.DataFrame(
        {
            "f0": [0, 0, 0, 1, 1, 1, 0, 1, 0, 1] * 10,
            "f1": np.random.default_rng(0).normal(size=100),
        }
    )
    y = pd.Series(X["f0"])

    tree = DecisionTreeClassifier(max_depth=2, random_state=42)
    tree.fit(X, y)

    ctx = _make_dummy_ctx(X, y, X, y, tree)
    shares = permutation_importance_share(ctx, n_repeats=3)

    assert "f0" in shares and "f1" in shares
    assert np.isclose(sum(shares.values()), 1.0)
    assert shares["f0"] > shares["f1"]
    assert shares["f0"] > 0.70


def test_leakage_identifier_s1_trigger_and_evaluate() -> None:
    """Feature with unique ratio >= 0.95 and high importance is supported."""
    n = 200
    # id_col has 195 unique values (97.5%)
    id_col = [f"id_{i}" for i in range(195)] + ["id_0", "id_1", "id_2", "id_3", "id_4"]
    y = np.array([0, 1] * 100)
    X = pd.DataFrame({"id_str": id_col, "noise": np.random.randn(n)})

    from sklearn.tree import DecisionTreeClassifier

    # Baseline model
    model = DecisionTreeClassifier(max_depth=3, random_state=42)
    model.fit(pd.DataFrame({"x": np.arange(n)}), y)

    # Mock predict so that permuting id_str drops accuracy substantially
    def mock_predict(df: pd.DataFrame) -> np.ndarray:
        # If id_str is original, return y; if permuted, return wrong
        if list(df["id_str"]) == id_col:
            return y
        return 1 - y

    mock_adapter = MagicMock()
    mock_adapter.predict.side_effect = mock_predict
    mock_adapter.predict_proba.return_value = np.column_stack([1 - y, y])

    ctx = _make_dummy_ctx(X, pd.Series(y), X, pd.Series(y), mock_adapter)
    hyp = LeakageIdentifierHypothesis()

    trig = hyp.trigger(ctx)
    assert trig.fired
    assert "id_str" in str(trig.values["flagged_features"])

    res = hyp.evaluate(ctx, trig, [])
    assert len(res) == 1
    hr = res[0]
    assert hr.subtype == "identifier"
    assert hr.status == Status.supported
    assert hr.evidence_strength == 1.0
    assert "Remove 'id_str'" in hr.recommended_actions[0]


def test_leakage_identifier_s2_row_order_trigger() -> None:
    """Row order strongly correlated with y, feature sorted by row -> S2 trigger fires."""
    n = 100
    # row index sorted: first 50 are 0, next 50 are 1 -> strong Spearman rho with y
    y = np.array([0] * 50 + [1] * 50)
    # row_id feature perfectly correlated with row_index
    row_feat = np.arange(n, dtype=float)
    X = pd.DataFrame({"row_id": row_feat, "feat": np.random.randn(n)})

    model = MagicMock()
    model.predict.return_value = y

    ctx = _make_dummy_ctx(
        X,
        pd.Series(y),
        X,
        pd.Series(y),
        model,
        orig_idx_tr=pd.Index(np.arange(n)),
        split_mode=SplitMode.single_file,
    )
    hyp = LeakageIdentifierHypothesis()

    trig = hyp.trigger(ctx)
    assert trig.fired
    assert "row_id" in str(trig.values["flagged_features"])


def test_leakage_identifier_s2_skipped_in_test_file_mode() -> None:
    """In test_file mode, row-order correlation S2 is skipped."""
    n = 100
    y = np.array([0] * 50 + [1] * 50)
    # Use genuinely continuous float values so S1 does not fire (fractional components present)
    rng = np.random.default_rng(99)
    cont_feat = rng.uniform(0.001, 1.0, size=n)  # all have fractional parts
    X = pd.DataFrame({"cont_feat": cont_feat})

    model = MagicMock()
    model.predict.return_value = y

    ctx = _make_dummy_ctx(
        X,
        pd.Series(y),
        X,
        pd.Series(y),
        model,
        split_mode=SplitMode.test_file,
    )
    hyp = LeakageIdentifierHypothesis()

    trig = hyp.trigger(ctx)
    assert not trig.fired
