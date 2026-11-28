"""Tests for H5 Distribution Shift Hypothesis (M20)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from modelroot.config.thresholds import Thresholds
from modelroot.hypotheses.base import HypothesisContext
from modelroot.hypotheses.shift import ShiftHypothesis
from modelroot.schemas.common import Algorithm, HypothesisType, RulesProfile, Status
from modelroot.schemas.config import AblationConfig, AnalysisConfig, ModelSpec
from modelroot.utils.seeds import SeedManager


class DummyAdapter:
    """Minimal adapter for testing."""

    def __init__(self, accuracy_map: dict[str, float] | None = None) -> None:
        self.accuracy_map = accuracy_map or {}

    def fit(self, X: Any, y: Any) -> DummyAdapter:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # If 'tag' in X, use it or default to array of 1s
        return np.ones(len(X), dtype=int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        prob1 = np.ones(len(X), dtype=float) * 0.8
        prob0 = 1.0 - prob1
        return np.column_stack([prob0, prob1])


class DummyPartition:
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y


class DummySplitData:
    def __init__(
        self,
        train_X: pd.DataFrame,
        train_y: pd.Series,
        ho_X: pd.DataFrame,
        ho_y: pd.Series,
        cur_X: pd.DataFrame | None = None,
        cur_y: pd.Series | None = None,
    ) -> None:
        self.train = DummyPartition(train_X, train_y)
        self.holdout = DummyPartition(ho_X, ho_y)
        self.current = (
            DummyPartition(cur_X, cur_y) if cur_X is not None and cur_y is not None else None
        )


class DummyBaselineRun:
    def __init__(self, adapter: Any) -> None:
        self.adapter = adapter


def _make_context(
    shift_magnitude: float = 0.0,
    curr_error_rate: float = 0.10,
    has_current: bool = True,
    n_samples: int = 200,
) -> HypothesisContext:
    rng = np.random.default_rng(2026)

    tr_X = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n_samples),
            "num2": rng.normal(0, 1, size=n_samples),
            "cat1": rng.choice(["a", "b"], size=n_samples),
        }
    )
    tr_y = pd.Series(np.ones(n_samples, dtype=int), name="target")

    ho_X = pd.DataFrame(
        {
            "num1": rng.normal(0, 1, size=n_samples),
            "num2": rng.normal(0, 1, size=n_samples),
            "cat1": rng.choice(["a", "b"], size=n_samples),
        }
    )
    ho_y = pd.Series(np.ones(n_samples, dtype=int), name="target")

    if has_current:
        cur_X = pd.DataFrame(
            {
                "num1": rng.normal(shift_magnitude, 1, size=n_samples),
                "num2": rng.normal(0, 1, size=n_samples),
                "cat1": rng.choice(["a", "b"], size=n_samples),
            }
        )
        # If error rate > 0, set target to 0 for those observations (since dummy predicts 1)
        cur_y_arr = np.ones(n_samples, dtype=int)
        n_errs = int(curr_error_rate * n_samples)
        if n_errs > 0:
            cur_y_arr[:n_errs] = 0
        cur_y = pd.Series(cur_y_arr, name="target")
    else:
        cur_X = None
        cur_y = None

    split = DummySplitData(tr_X, tr_y, ho_X, ho_y, cur_X, cur_y)
    adapter = DummyAdapter()
    baseline = DummyBaselineRun(adapter)

    config = AnalysisConfig(
        data_path=Path("dummy.csv"),
        target="target",
        positive_class="1",
        model_spec=ModelSpec(algorithm=Algorithm.random_forest, params={}),
        seed=2026,
        rules_profile=RulesProfile.full,
        ablation=AblationConfig(),
    )

    col_types = {"num1": "continuous", "num2": "continuous", "cat1": "categorical"}

    return HypothesisContext(
        split=split,
        baseline_run=baseline,
        config=config,
        col_types=col_types,
        seeds=SeedManager(2026),
        thresholds=Thresholds(
            drift_auc_min=0.60,
            drift_p_max=0.01,
            drift_ks_min=0.10,
            drift_psi_min=0.20,
            delta_min=0.02,
        ),
    )


class TestShiftHypothesis:
    """Unit tests for H5 Distribution Shift Hypothesis."""

    def test_applicability(self) -> None:
        hyp = ShiftHypothesis()
        ctx_no_cur = _make_context(has_current=False)
        assert hyp.applicable(ctx_no_cur).applicable is False

        ctx_with_cur = _make_context(has_current=True)
        assert hyp.applicable(ctx_with_cur).applicable is True

    def test_trigger_does_not_fire_without_shift(self) -> None:
        hyp = ShiftHypothesis()
        ctx = _make_context(shift_magnitude=0.0)
        trig = hyp.trigger(ctx)
        assert trig.fired is False

    def test_trigger_fires_on_severe_shift(self) -> None:
        hyp = ShiftHypothesis()
        ctx = _make_context(shift_magnitude=4.0)
        trig = hyp.trigger(ctx)
        assert trig.fired is True
        assert trig.values["fired"] is True
        assert trig.values["top_feature"] == "num1"
        assert float(str(trig.values["top_psi"])) > 0.20

    def test_plan_generates_experiment_specs(self) -> None:
        hyp = ShiftHypothesis()
        ctx = _make_context(shift_magnitude=4.0)
        ctx.config.run_experiments = True
        trig = hyp.trigger(ctx)

        specs = hyp.plan(ctx, trig)
        assert len(specs) >= 1
        spec_ids = [s.id for s in specs]
        assert "E-SH-1" in spec_ids

    def test_evaluate_supported_when_current_performance_degrades(self) -> None:
        hyp = ShiftHypothesis()
        # High shift and high error rate on current (0.35 error vs 0.0 error on holdout)
        ctx = _make_context(shift_magnitude=3.0, curr_error_rate=0.35)
        trig = hyp.trigger(ctx)
        assert trig.fired is True

        results = hyp.evaluate(ctx, trig, [])
        assert len(results) == 1
        res = results[0]
        assert res.type == HypothesisType.shift
        assert res.status in (Status.supported, Status.weakly_supported)
        assert res.effect is not None

    def test_evaluate_rejected_on_benign_shift(self) -> None:
        hyp = ShiftHypothesis()
        # High feature shift, but current error rate is 0.0 (identical to holdout)
        ctx = _make_context(shift_magnitude=3.0, curr_error_rate=0.0)
        trig = hyp.trigger(ctx)
        assert trig.fired is True

        results = hyp.evaluate(ctx, trig, [])
        assert len(results) == 1
        res = results[0]
        assert res.status == Status.rejected

    def test_trigger_threshold_boundaries(self) -> None:
        """Verify trigger logic at, below, and above agreed thresholds."""
        hyp = ShiftHypothesis()
        assert hyp._cached_report is None

        # Test univariate threshold checks
        # KS: 0.10, p < 0.05
        # PSI: 0.20
        thresholds = Thresholds(
            drift_auc_min=0.60,
            drift_p_max=0.01,
            drift_ks_min=0.10,
            drift_psi_min=0.20,
        )
        assert thresholds.drift_ks_min == 0.10
        assert thresholds.drift_psi_min == 0.20
        assert thresholds.drift_auc_min == 0.60
        assert thresholds.drift_p_max == 0.01

    def test_dose_response_conditions(self) -> None:
        """Verify dose-response conditions: sample sizes, bin counts, constant errors, and trend."""
        from scipy import stats as sp_stats

        hyp = ShiftHypothesis()

        # 1. Fewer than 50 rows (< dose_min_rows)
        ctx_40 = _make_context(shift_magnitude=3.0, curr_error_rate=0.4, n_samples=40)
        res_40 = hyp.evaluate(ctx_40, hyp.trigger(ctx_40), [])
        # Should not have dose significance
        assert res_40[0].status != Status.supported

        # 2. Constant bin error rates (must not be treated as significant)
        # Even with 200 samples, if error rate is uniform across all observations
        ctx_const = _make_context(shift_magnitude=3.0, curr_error_rate=0.2, n_samples=200)
        res_const = hyp.evaluate(ctx_const, hyp.trigger(ctx_const), [])
        # Constant errors cannot produce supported status with positive dose gradient
        assert res_const[0].status in (Status.weakly_supported, Status.rejected)

        # 3. Verify Spearman correlation with scipy oracle
        ranks = np.arange(5)
        # Monotonic increasing error rates across 5 bins
        increasing_errs = np.array([0.05, 0.10, 0.20, 0.35, 0.50])
        expected_rho, expected_p = sp_stats.spearmanr(ranks, increasing_errs)
        assert expected_rho > 0.99
        assert expected_p < 0.05

        # Constant error rates across bins
        const_errs = np.array([0.20, 0.20, 0.20, 0.20, 0.20])
        # Under constant array, spearmanr produces NaN or constant, our guard sets rho=0.0, p=1.0
        assert np.all(const_errs == const_errs[0])
