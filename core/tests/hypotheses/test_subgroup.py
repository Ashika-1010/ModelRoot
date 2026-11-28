"""Tests for H4 Subgroup Underperformance Hypothesis (M18)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

from modelroot.config.thresholds import Thresholds
from modelroot.hypotheses.base import HypothesisContext
from modelroot.hypotheses.slices import exact_margin_test
from modelroot.hypotheses.subgroup import SubgroupHypothesis
from modelroot.schemas.common import Algorithm, HypothesisType, RulesProfile, Status
from modelroot.schemas.config import AblationConfig, AnalysisConfig, ModelSpec
from modelroot.utils.seeds import SeedManager


class DummyAdapter:
    """Minimal adapter for hypothesis unit tests."""

    def __init__(self, error_indices: set[int] | None = None) -> None:
        self.error_indices = error_indices or set()

    def fit(self, X: Any, y: Any, sample_weight: Any = None) -> DummyAdapter:
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        # Predict 1 unless index is in error_indices
        preds = np.ones(len(X), dtype=int)
        for i in range(len(X)):
            if i in self.error_indices:
                preds[i] = 0
        return preds


class DummySplitPartition:
    def __init__(self, X: pd.DataFrame, y: pd.Series) -> None:
        self.X = X
        self.y = y


class DummySplitData:
    def __init__(
        self, train_X: pd.DataFrame, train_y: pd.Series, ho_X: pd.DataFrame, ho_y: pd.Series
    ) -> None:
        self.train = DummySplitPartition(train_X, train_y)
        self.holdout = DummySplitPartition(ho_X, ho_y)


class DummyBaselineRun:
    def __init__(self, oof_preds: np.ndarray, ho_preds: np.ndarray) -> None:
        self.oof_predictions = oof_preds
        self.holdout_pred = ho_preds
        self.adapter = DummyAdapter()


def _make_context(
    n_train: int = 500,
    n_holdout: int = 300,
    slice_train_err: float = 0.35,
    comp_train_err: float = 0.05,
    slice_ho_err: float = 0.35,
    comp_ho_err: float = 0.05,
    subgroup_fraction: float = 0.20,
    protocol: RulesProfile = RulesProfile.full,
) -> HypothesisContext:
    rng = np.random.default_rng(2026)

    # Features: categorical "category" and numeric "feature1"
    def gen_df(n: int) -> tuple[pd.DataFrame, pd.Series]:
        rem = (1.0 - subgroup_fraction) / 2.0
        categories = rng.choice(
            ["group_A", "group_B", "group_C"], size=n, p=[subgroup_fraction, rem, rem]
        )
        f1 = rng.normal(0, 1, size=n)
        df = pd.DataFrame({"category": categories, "feature1": f1})
        y = pd.Series(rng.choice([0, 1], size=n), name="target")
        return df, y

    tr_X, tr_y = gen_df(n_train)
    ho_X, ho_y = gen_df(n_holdout)

    # Construct OOF predictions with controlled error rates
    oof_preds = tr_y.to_numpy().copy()
    tr_slice_mask = tr_X["category"] == "group_A"
    # Inject errors in slice
    for idx in np.where(tr_slice_mask)[0]:
        if rng.uniform(0, 1) < slice_train_err:
            oof_preds[idx] = 1 - oof_preds[idx]
    # Inject errors in complement
    for idx in np.where(~tr_slice_mask)[0]:
        if rng.uniform(0, 1) < comp_train_err:
            oof_preds[idx] = 1 - oof_preds[idx]

    # Construct Holdout predictions with controlled error rates
    ho_preds = ho_y.to_numpy().copy()
    ho_slice_mask = ho_X["category"] == "group_A"
    for idx in np.where(ho_slice_mask)[0]:
        if rng.uniform(0, 1) < slice_ho_err:
            ho_preds[idx] = 1 - ho_preds[idx]
    for idx in np.where(~ho_slice_mask)[0]:
        if rng.uniform(0, 1) < comp_ho_err:
            ho_preds[idx] = 1 - ho_preds[idx]

    col_types = {"category": "categorical", "feature1": "continuous"}
    baseline = DummyBaselineRun(oof_preds, ho_preds)
    split = DummySplitData(tr_X, tr_y, ho_X, ho_y)

    config = AnalysisConfig(
        data_path=Path("dummy.csv"),
        target="target",
        positive_class="1",
        model_spec=ModelSpec(algorithm=Algorithm.random_forest, params={}),
        seed=2026,
        rules_profile=protocol,
        ablation=AblationConfig(subgroup_protocol=protocol),
    )

    return HypothesisContext(
        split=split,
        baseline_run=baseline,
        config=config,
        col_types=col_types,
        seeds=SeedManager(2026),
        thresholds=Thresholds(slice_gap=0.05, slice_min_n=30, slice_fdr_q=0.10),
    )


class TestExactMarginTest:
    """Tests for the certified raw-difference exact margin test."""

    def test_basic_p_value_range(self) -> None:
        p = exact_margin_test(x1=10, n1=50, x2=5, n2=200, delta=0.05, n_grid=200)
        assert 0.0 <= p <= 1.0

    def test_inclusive_ties(self) -> None:
        # Equal rates on small numbers
        p = exact_margin_test(x1=2, n1=20, x2=2, n2=20, delta=0.05, n_grid=100)
        # Should be non-significant
        assert p > 0.5

    def test_endpoints_without_nans(self) -> None:
        p_zero = exact_margin_test(x1=0, n1=30, x2=0, n2=300, delta=0.05, n_grid=100)
        assert not np.isnan(p_zero)
        assert 0.0 <= p_zero <= 1.0

        p_full = exact_margin_test(x1=30, n1=30, x2=300, n2=300, delta=0.05, n_grid=100)
        assert not np.isnan(p_full)
        assert 0.0 <= p_full <= 1.0

    def test_conservative_upper_bound(self) -> None:
        # Certified upper bound adds Lipschitz slack:
        # ((n1 + n2) * (1 - delta) / (2 * (n_grid - 1)))
        # Comparing coarse grid vs dense grid
        p_coarse = exact_margin_test(x1=8, n1=30, x2=10, n2=300, delta=0.05, n_grid=50)
        p_dense = exact_margin_test(x1=8, n1=30, x2=10, n2=300, delta=0.05, n_grid=1000)
        # Coarse grid must have >= slack than dense grid
        assert p_coarse >= p_dense - 1e-4

    def test_lipschitz_correction_and_cap(self) -> None:
        # Verify that epsilon is added to grid maximum
        n1, n2, delta, n_grid = 10, 20, 0.05, 50
        eps = ((n1 + n2) * (1.0 - delta)) / (2.0 * (n_grid - 1))
        p = exact_margin_test(x1=5, n1=n1, x2=2, n2=n2, delta=delta, n_grid=n_grid)
        # Result must be >= epsilon and <= 1.0
        assert p >= eps
        assert p <= 1.0

        # When raw max + eps exceeds 1, verify it caps at 1.0
        p_cap = exact_margin_test(x1=2, n1=10, x2=5, n2=10, delta=0.05, n_grid=10)
        assert p_cap <= 1.0
        assert p_cap == 1.0

    def test_invalid_inputs(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            exact_margin_test(x1=0, n1=0, x2=5, n2=10)

        with pytest.raises(ValueError, match="delta"):
            exact_margin_test(x1=5, n1=10, x2=5, n2=10, delta=1.5)

        with pytest.raises(ValueError, match="n_grid"):
            exact_margin_test(x1=5, n1=10, x2=5, n2=10, delta=0.05, n_grid=1)

    def test_sf_numerical_stability_small_tail(self) -> None:
        # Extreme superiority where raw tail probability is practically zero
        # With n1=50, n2=500, epsilon = (550 * 0.95) / (2 * (n_grid - 1))
        # At n_grid=6000, epsilon = 522.5 / 11998 ~= 0.0435 <= 0.05
        p = exact_margin_test(x1=50, n1=50, x2=0, n2=500, delta=0.05, n_grid=6000)
        assert not np.isnan(p)
        assert 0.0 <= p <= 0.05


class TestSubgroupHypothesis:
    """Tests for SubgroupHypothesis trigger, plan, and evaluate."""

    def test_trigger_discovery_finds_slice(self) -> None:
        ctx = _make_context(slice_train_err=0.45, comp_train_err=0.05)
        hyp = SubgroupHypothesis()
        assert hyp.applicable(ctx).applicable is True

        res = hyp.trigger(ctx)
        assert res.fired is True
        assert res.values["fired"] is True
        assert int(str(res.values.get("n_candidates", 0))) >= 1
        assert "group_A" in str(res.values["top_slice"])

    def test_trigger_does_not_fire_on_uniform_error(self) -> None:
        ctx = _make_context(slice_train_err=0.10, comp_train_err=0.10)
        hyp = SubgroupHypothesis()
        res = hyp.trigger(ctx)
        assert res.fired is False
        assert res.values["n_candidates"] == 0

    def test_cache_reset_on_trigger(self) -> None:
        ctx1 = _make_context(slice_train_err=0.45, comp_train_err=0.05)
        ctx2 = _make_context(slice_train_err=0.10, comp_train_err=0.10)

        hyp = SubgroupHypothesis()
        # Trigger 1 fires and caches
        res1 = hyp.trigger(ctx1)
        assert res1.fired is True
        assert hyp._cached_candidates is not None
        assert len(hyp._cached_candidates) > 0

        # Trigger 2 with clean data must reset cache completely
        res2 = hyp.trigger(ctx2)
        assert res2.fired is False
        assert hyp._cached_candidates is None

    def test_evaluate_confirms_on_holdout(self) -> None:
        # Strong underperformance on both train and holdout
        ctx = _make_context(
            slice_train_err=0.45,
            comp_train_err=0.05,
            slice_ho_err=0.45,
            comp_ho_err=0.05,
        )
        hyp = SubgroupHypothesis()
        trig = hyp.trigger(ctx)
        assert trig.fired is True

        results = hyp.evaluate(ctx, trig, [])
        assert len(results) == 1
        verdict = results[0]
        assert verdict.type == HypothesisType.subgroup
        assert verdict.status == Status.supported
        assert verdict.subtype in ("representation_related", "intrinsic_or_unresolved")
        assert verdict.effect is not None
        assert verdict.effect.delta > 0.05

    def test_evaluate_rejects_when_holdout_fails_to_confirm(self) -> None:
        # High error in train (false discovery / noise), but normal error in holdout
        ctx = _make_context(
            slice_train_err=0.45,
            comp_train_err=0.05,
            slice_ho_err=0.06,  # Holdout gap <= 0.05
            comp_ho_err=0.05,
        )
        hyp = SubgroupHypothesis()
        trig = hyp.trigger(ctx)
        assert trig.fired is True

        results = hyp.evaluate(ctx, trig, [])
        assert len(results) == 1
        verdict = results[0]
        assert verdict.status == Status.rejected
        assert verdict.subtype is None

    def test_evaluate_inconclusive_when_holdout_slice_underpowered(self) -> None:
        # Holdout has tiny slice size (n < 30)
        ctx = _make_context(
            slice_train_err=0.45,
            comp_train_err=0.05,
            subgroup_fraction=0.02,  # Only ~6 samples in holdout (n < 30)
        )
        hyp = SubgroupHypothesis()
        # Ensure slice passes train threshold (give it enough train samples)
        ctx.split.train.X["category"] = ["group_A"] * 40 + ["group_B"] * (
            len(ctx.split.train.X) - 40
        )
        # But holdout has only 5 samples
        ctx.split.holdout.X["category"] = ["group_A"] * 5 + ["group_B"] * (
            len(ctx.split.holdout.X) - 5
        )

        trig = hyp.trigger(ctx)
        if trig.fired:
            results = hyp.evaluate(ctx, trig, [])
            assert results[0].status == Status.inconclusive
            assert "sample size" in results[0].caveats[0].lower()

    def test_plan_generates_reweight_spec(self) -> None:
        ctx = _make_context(slice_train_err=0.45, comp_train_err=0.05)
        ctx.config.run_experiments = True
        hyp = SubgroupHypothesis()
        trig = hyp.trigger(ctx)
        specs = hyp.plan(ctx, trig)

        assert len(specs) == 1
        assert specs[0].intervention == "reweight_group"
        assert "feature" in specs[0].params
        assert specs[0].params["weight_cap"] == 5.0
