"""Tests for M16 Leakage hypothesis: Target-Proxy (H3c)."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from modelroot.assessment.baseline import BaselineRun, PredictionSet
from modelroot.config.thresholds import Thresholds
from modelroot.hypotheses.base import HypothesisContext
from modelroot.hypotheses.leakage import (
    LeakageTargetProxyHypothesis,
    flip_effect,
    single_feature_score,
    status_for_proxy,
)
from modelroot.schemas.common import HypothesisType, PrimaryMetric, RulesProfile, SplitMode, Status
from modelroot.stats.bootstrap import BootstrapResult
from modelroot.utils.seeds import SeedManager

# ---------------------------------------------------------------------------
# Shared helper
# ---------------------------------------------------------------------------


def _make_dummy_ctx(
    X_tr: pd.DataFrame,
    y_tr: pd.Series,
    X_ho: pd.DataFrame,
    y_ho: pd.Series,
    fitted_model: Any,
    X_current: pd.DataFrame | None = None,
    y_current: pd.Series | None = None,
    orig_idx_tr: pd.Index | None = None,
    split_mode: SplitMode = SplitMode.single_file,
    metric: PrimaryMetric = PrimaryMetric.accuracy,
    unavailable_at_inference: list[str] | None = None,
    budget_check: Any = None,
) -> HypothesisContext:
    split = MagicMock()
    split.X_train = X_tr
    split.y_train = y_tr
    split.X_holdout = X_ho
    split.y_holdout = y_ho
    split.X_current = X_current
    split.y_current = y_current
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
    current_pred = None
    if X_current is not None and y_current is not None:
        y_curr_pred = (
            fitted_model.predict(X_current)
            if hasattr(fitted_model, "predict")
            else np.zeros(len(y_current), dtype=int)
        )
        y_curr_proba = (
            fitted_model.predict_proba(X_current)[:, 1]
            if hasattr(fitted_model, "predict_proba") and not isinstance(fitted_model, MagicMock)
            else np.zeros(len(y_current), dtype=float)
        )
        current_pred = PredictionSet(
            index=X_current.index,
            y_true=y_current.to_numpy(),
            y_pred=y_curr_pred,
            y_proba=y_curr_proba,
        )

    baseline_run = BaselineRun(
        adapter=fitted_model,
        train_pred=train_pred,
        holdout_pred=holdout_pred,
        current_pred=current_pred,
        metrics=[],
    )

    cfg = MagicMock()
    cfg.primary_metric = metric
    cfg.rules_profile = RulesProfile.full
    cfg.split = MagicMock()
    cfg.split.mode = split_mode
    cfg.model_spec = {"algorithm": "decision_tree", "params": {"max_depth": 3}}
    cfg.seed = 42
    cfg.budget_s = 60.0

    # metadata as real dictionary
    cfg.metadata = {"unavailable_at_inference": unavailable_at_inference or []}

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


def _make_trained_model(X_tr: pd.DataFrame, y_tr: np.ndarray) -> Any:
    """Train a small LogisticRegression for use as fitted adapter."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=200, random_state=0)
    clf.fit(X_tr, y_tr)
    return clf


def _synth_data(
    n: int = 400, n_features: int = 5, seed: int = 0
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, n_features))
    y = (X[:, 0] > 0).astype(int)
    cols = [f"f{i}" for i in range(n_features)]
    return pd.DataFrame(X, columns=cols), y


# ---------------------------------------------------------------------------
# single_feature_score
# ---------------------------------------------------------------------------


def test_single_feature_score_informative() -> None:
    """An informative feature should yield AUC > 0.5."""
    n = 300
    rng = np.random.default_rng(1)
    y = rng.integers(0, 2, size=n)
    X = pd.DataFrame({"f": y.astype(float) + rng.standard_normal(n) * 0.1})
    score = single_feature_score("f", X, y, cv=3, seed=0)
    assert score > 0.80, f"Expected high AUC for informative feature; got {score:.4f}"


def test_single_feature_score_uninformative() -> None:
    """A pure-noise feature should yield AUC close to 0.5."""
    n = 300
    rng = np.random.default_rng(2)
    y = rng.integers(0, 2, size=n)
    X = pd.DataFrame({"f": rng.standard_normal(n)})
    score = single_feature_score("f", X, y, cv=3, seed=0)
    # Noise features produce near-chance AUC; allow some sampling variance
    assert score < 0.70, f"Expected near-chance AUC for noise feature; got {score:.4f}"


def test_single_feature_score_single_class_returns_half() -> None:
    """When only one class is present, return 0.5."""
    n = 50
    X = pd.DataFrame({"f": np.ones(n)})
    y = np.zeros(n, dtype=int)
    score = single_feature_score("f", X, y, cv=3, seed=0)
    assert score == 0.5


# ---------------------------------------------------------------------------
# flip_effect
# ---------------------------------------------------------------------------


def test_flip_effect_inverts_direction() -> None:
    boot = BootstrapResult(
        delta=0.10,
        ci_low=0.02,
        ci_high=0.18,
        se=0.04,
        n_valid=900,
        n_boot=1000,
        metric_a=0.70,
        metric_b=0.80,
    )
    flipped = flip_effect(boot)
    assert abs(flipped.delta - (-0.10)) < 1e-9
    assert abs(flipped.ci_low - (-0.18)) < 1e-9
    assert abs(flipped.ci_high - (-0.02)) < 1e-9
    assert abs(flipped.metric_a - 0.80) < 1e-9
    assert abs(flipped.metric_b - 0.70) < 1e-9
    assert flipped.se == boot.se
    assert flipped.n_valid == boot.n_valid


def test_flip_effect_double_inversion_identity() -> None:
    boot = BootstrapResult(
        delta=0.05,
        ci_low=-0.01,
        ci_high=0.11,
        se=0.03,
        n_valid=800,
        n_boot=1000,
        metric_a=0.60,
        metric_b=0.65,
    )
    double = flip_effect(flip_effect(boot))
    assert abs(double.delta - boot.delta) < 1e-9
    assert abs(double.ci_low - boot.ci_low) < 1e-9
    assert abs(double.ci_high - boot.ci_high) < 1e-9


# ---------------------------------------------------------------------------
# status_for_proxy
# ---------------------------------------------------------------------------


def test_status_for_proxy_flagged_high_share_gives_supported() -> None:
    """Rule 1: user-flagged with share >= 0.10 → supported."""
    t = Thresholds()
    s = status_for_proxy(
        generic_status=Status.weakly_supported,
        suspects=["proxy"],
        user_flagged={"proxy"},
        shares={"proxy": 0.15},
        transfer_supported=False,
        thresholds=t,
    )
    assert s == Status.supported


def test_status_for_proxy_transfer_gives_supported() -> None:
    """Rule 2: transfer_supported → supported."""
    t = Thresholds()
    s = status_for_proxy(
        generic_status=Status.weakly_supported,
        suspects=["proxy"],
        user_flagged=set(),
        shares={"proxy": 0.0},
        transfer_supported=True,
        thresholds=t,
    )
    assert s == Status.supported


def test_status_for_proxy_all_flagged_low_share_gives_rejected() -> None:
    """Rule 3: every suspect user-flagged AND every share < 0.05 → rejected."""
    t = Thresholds()
    s = status_for_proxy(
        generic_status=Status.supported,
        suspects=["a", "b"],
        user_flagged={"a", "b"},
        shares={"a": 0.03, "b": 0.01},
        transfer_supported=False,
        thresholds=t,
    )
    assert s == Status.rejected


def test_status_for_proxy_generic_supported_capped_at_weakly() -> None:
    """Rule 4: generic supported → capped at weakly_supported when no stronger evidence."""
    t = Thresholds()
    s = status_for_proxy(
        generic_status=Status.supported,
        suspects=["a"],
        user_flagged=set(),
        shares={"a": 0.30},
        transfer_supported=False,
        thresholds=t,
    )
    assert s == Status.weakly_supported


def test_status_for_proxy_weakly_supported_passes_through() -> None:
    """weakly_supported passes through rule 4 unchanged."""
    t = Thresholds()
    s = status_for_proxy(
        generic_status=Status.weakly_supported,
        suspects=["a"],
        user_flagged=set(),
        shares={"a": 0.0},
        transfer_supported=False,
        thresholds=t,
    )
    assert s == Status.weakly_supported


def test_status_for_proxy_flagged_low_share_no_reject_if_not_all_flagged() -> None:
    """Rule 3 requires ALL suspects to be user-flagged; if only some → cap, not reject."""
    t = Thresholds()
    s = status_for_proxy(
        generic_status=Status.supported,
        suspects=["a", "b"],
        user_flagged={"a"},  # only one is flagged
        shares={"a": 0.01, "b": 0.01},
        transfer_supported=False,
        thresholds=t,
    )
    # generic=supported → rule 4 caps at weakly_supported
    assert s == Status.weakly_supported


def test_status_for_proxy_rule1_beats_rule3() -> None:
    """Rule 1 (flagged high share → supported) has priority over rule 3."""
    t = Thresholds()
    s = status_for_proxy(
        generic_status=Status.supported,
        suspects=["a", "b"],
        user_flagged={"a", "b"},
        # 'a' has high share (rule 1), 'b' low (rule 3 not applicable alone)
        shares={"a": 0.20, "b": 0.01},
        transfer_supported=False,
        thresholds=t,
    )
    assert s == Status.supported


# ---------------------------------------------------------------------------
# Property test: status cap invariant
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", range(20))
def test_status_for_proxy_cap_invariant(seed: int) -> None:
    """
    Property: when neither rule 1 nor rule 2 applies,
    status_for_proxy(supported) must never return Status.supported.
    """
    rng = np.random.default_rng(seed)
    t = Thresholds()

    n_suspects = rng.integers(1, 4)
    suspects = [f"f{i}" for i in range(n_suspects)]
    # Shares in [0, 0.09] - low enough that rule 1 won't fire
    shares = {s: float(rng.uniform(0, 0.09)) for s in suspects}
    # Only some are user-flagged
    user_flagged = {s for s in suspects if rng.random() < 0.5}

    result = status_for_proxy(
        generic_status=Status.supported,
        suspects=suspects,
        user_flagged=user_flagged,
        shares=shares,
        transfer_supported=False,
        thresholds=t,
    )
    # Must NOT return supported (rule 4 caps it)
    # Unless rule 3 fires first (all flagged, all shares < 0.05)
    all_flagged = all(s in user_flagged for s in suspects)
    all_low = all(shares.get(s, 0.0) < t.flagged_imp_min for s in suspects)
    if all_flagged and all_low:
        assert result == Status.rejected
    else:
        assert result != Status.supported


# ---------------------------------------------------------------------------
# H3c not_triggered when no suspect features
# ---------------------------------------------------------------------------


def test_target_proxy_not_triggered_on_clean_data() -> None:
    """H3c returns not_triggered on clean uncorrelated features."""
    rng = np.random.default_rng(99)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 5)), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.standard_normal(n) > 0).astype(int))

    n_tr = 200
    X_tr, y_tr = X.iloc[:n_tr].reset_index(drop=True), y.iloc[:n_tr].reset_index(drop=True)
    X_ho, y_ho = X.iloc[n_tr:].reset_index(drop=True), y.iloc[n_tr:].reset_index(drop=True)

    model = _make_trained_model(X_tr, y_tr.to_numpy())
    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model)

    hyp = LeakageTargetProxyHypothesis()
    trigger = hyp.trigger(ctx)

    # Clean random data should not trigger (all feature AUCs near 0.5, low importance)
    # This is a probabilistic test; with random seed 99 and n=300, spurious triggers
    # are extremely unlikely. We allow weak_support but not supported.
    results = hyp.evaluate(ctx, trigger, [])
    assert len(results) == 1
    assert results[0].type == HypothesisType.leakage
    assert results[0].subtype == "target_proxy"
    # Status must not be "supported" on clean uncorrelated data
    assert results[0].status != Status.supported


# ---------------------------------------------------------------------------
# H3c detects near-perfect proxy
# ---------------------------------------------------------------------------


def test_target_proxy_detected_with_proxy_feature() -> None:
    """H3c triggers and returns weakly_supported or higher with a noisy proxy of y."""
    rng = np.random.default_rng(7)
    n = 500
    # 4 noise features + 1 proxy = noisy y
    X_noise = rng.standard_normal((n, 4))
    y_arr = (rng.standard_normal(n) > 0).astype(int)
    proxy = y_arr.astype(float) + rng.standard_normal(n) * 0.2

    X = pd.DataFrame(np.column_stack([X_noise, proxy]), columns=["n0", "n1", "n2", "n3", "proxy"])
    y = pd.Series(y_arr)

    n_tr = 350
    X_tr, y_tr = X.iloc[:n_tr].reset_index(drop=True), y.iloc[:n_tr].reset_index(drop=True)
    X_ho, y_ho = X.iloc[n_tr:].reset_index(drop=True), y.iloc[n_tr:].reset_index(drop=True)

    model = _make_trained_model(X_tr, y_tr.to_numpy())
    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model)

    hyp = LeakageTargetProxyHypothesis()
    trigger = hyp.trigger(ctx)

    # proxy should be in suspects
    suspects_str = str(trigger.values.get("suspects") or "")
    suspects = [s for s in suspects_str.split(",") if s]
    assert "proxy" in suspects, f"Expected 'proxy' in suspects; got {suspects}"
    assert trigger.fired


# ---------------------------------------------------------------------------
# H3c user-flagged suspect
# ---------------------------------------------------------------------------


def test_target_proxy_user_flagged_in_suspects() -> None:
    """User-flagged features appear in suspects regardless of AUC."""
    rng = np.random.default_rng(13)
    n = 400
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "unavail"])
    y = pd.Series((X["a"].to_numpy() > 0).astype(int))

    n_tr = 280
    X_tr, y_tr = X.iloc[:n_tr].reset_index(drop=True), y.iloc[:n_tr].reset_index(drop=True)
    X_ho, y_ho = X.iloc[n_tr:].reset_index(drop=True), y.iloc[n_tr:].reset_index(drop=True)

    model = _make_trained_model(X_tr, y_tr.to_numpy())
    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model, unavailable_at_inference=["unavail"])

    hyp = LeakageTargetProxyHypothesis()
    trigger = hyp.trigger(ctx)

    suspects_str = str(trigger.values.get("suspects") or "")
    suspects = [s for s in suspects_str.split(",") if s]
    assert "unavail" in suspects, f"Expected 'unavail' in suspects due to user flag; got {suspects}"
    assert trigger.fired


# ---------------------------------------------------------------------------
# H3c verbatim caveat
# ---------------------------------------------------------------------------


def test_target_proxy_verbatim_caveat_present() -> None:
    """The mandatory caveat must appear in every result."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(rng.standard_normal((n, 3)), columns=["a", "b", "c"])
    y = pd.Series((X["a"].to_numpy() > 0).astype(int))

    n_tr = 200
    X_tr, y_tr = X.iloc[:n_tr].reset_index(drop=True), y.iloc[:n_tr].reset_index(drop=True)
    X_ho, y_ho = X.iloc[n_tr:].reset_index(drop=True), y.iloc[n_tr:].reset_index(drop=True)

    model = _make_trained_model(X_tr, y_tr.to_numpy())
    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model)

    hyp = LeakageTargetProxyHypothesis()
    trigger = hyp.trigger(ctx)
    results = hyp.evaluate(ctx, trigger, [])

    expected_caveat = (
        "ModelRoot cannot distinguish proxy leakage from a legitimately strong feature "
        "without knowing whether the feature is available at prediction time; "
        "this finding requires domain review."
    )
    for r in results:
        assert any(expected_caveat in c for c in r.caveats), (
            f"Verbatim caveat missing; found: {r.caveats}"
        )


# ---------------------------------------------------------------------------
# H3c suspect sorting: user-flagged first, then AUC descending, alphabetical ties
# ---------------------------------------------------------------------------


def test_target_proxy_suspect_ordering() -> None:
    """
    Suspects: user-flagged comes first, then AUC-descending, then alpha.
    Test by patching single_feature_score.
    """
    rng = np.random.default_rng(99)
    n = 400
    X = pd.DataFrame(rng.standard_normal((n, 4)), columns=["alpha", "beta", "flagged", "delta"])
    y = pd.Series((X["alpha"].to_numpy() > 0).astype(int))

    n_tr = 280
    X_tr, y_tr = X.iloc[:n_tr].reset_index(drop=True), y.iloc[:n_tr].reset_index(drop=True)
    X_ho, y_ho = X.iloc[n_tr:].reset_index(drop=True), y.iloc[n_tr:].reset_index(drop=True)

    model = _make_trained_model(X_tr, y_tr.to_numpy())
    ctx = _make_dummy_ctx(X_tr, y_tr, X_ho, y_ho, model, unavailable_at_inference=["flagged"])

    # Patch single_feature_score to return controlled AUC values
    auc_map = {"alpha": 0.95, "beta": 0.92, "flagged": 0.70, "delta": 0.91}

    def mock_sfs(feature: str, X: pd.DataFrame, y: np.ndarray, **kwargs: Any) -> float:
        return auc_map.get(feature, 0.5)

    hyp = LeakageTargetProxyHypothesis()
    with patch("modelroot.hypotheses.leakage.single_feature_score", side_effect=mock_sfs):
        trigger = hyp.trigger(ctx)

    suspects_str = str(trigger.values.get("suspects") or "")
    suspects = [s for s in suspects_str.split(",") if s]

    # flagged first (user flag), then alpha (0.95 > beta 0.92 > delta 0.91)
    # Only MAX_SUSPECTS=3 are kept
    assert suspects[0] == "flagged", f"Expected flagged first; got {suspects}"
    assert suspects[1] == "alpha", f"Expected alpha second; got {suspects}"
    assert suspects[2] == "beta", f"Expected beta third; got {suspects}"
    assert len(suspects) == 3
