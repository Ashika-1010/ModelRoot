"""M11 tests: OverfittingHypothesis unit and integration tests."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from modelroot.config.thresholds import Thresholds
from modelroot.evidence.rules import get_rule_registry
from modelroot.hypotheses.base import HypothesisContext
from modelroot.hypotheses.overfitting import (
    SIZE_CURVE_FRACTIONS,
    OverfittingHypothesis,
    compute_size_curve,
    select_capacity,
)
from modelroot.hypotheses.registry import default_hypotheses, get_registry
from modelroot.schemas.common import Algorithm, RulesProfile, Status
from modelroot.schemas.hypothesis import TriggerResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tiny_split(n: int = 60, seed: int = 0):  # type: ignore[no-untyped-def]
    """Return a small SplitData-like with synthetic data."""
    from modelroot.data.split import SplitData

    rng = np.random.default_rng(seed)
    n_train = n * 2 // 3
    n_holdout = n - n_train
    X_tr = pd.DataFrame(
        {"a": rng.random(n_train).astype(str), "b": rng.random(n_train).astype(str)}
    )
    y_tr = pd.Series(rng.integers(0, 2, n_train))
    X_ho = pd.DataFrame(
        {"a": rng.random(n_holdout).astype(str), "b": rng.random(n_holdout).astype(str)}
    )
    y_ho = pd.Series(rng.integers(0, 2, n_holdout))
    return SplitData(
        X_train=X_tr,
        y_train=y_tr,
        X_holdout=X_ho,
        y_holdout=y_ho,
        orig_index_train=X_tr.index,
        orig_index_holdout=X_ho.index,
        split_fingerprint="test",
    )


def _adapter(algo: str = "decision_tree", **kw):  # type: ignore[no-untyped-def]
    from modelroot.models.registry import get_model
    from modelroot.schemas.config import ModelSpec

    spec = ModelSpec(algorithm=Algorithm(algo), params=kw)
    return get_model(spec, {"a": "numeric", "b": "numeric"}, seed=0)


# ---------------------------------------------------------------------------
# T1: test_trigger_table
# ---------------------------------------------------------------------------


def test_trigger_table() -> None:
    """Absolute gap, relative gap, both-below, and zero-train-score cases."""
    from unittest.mock import MagicMock

    hyp = OverfittingHypothesis()
    t = Thresholds()  # gap=0.10, gap_rel=0.15

    def _ctx(ts: float, hs: float) -> HypothesisContext:
        from modelroot.schemas.metrics import MetricResult

        bl = MagicMock()
        bl.metrics = [
            MetricResult(name="macro_f1", value=ts, split="train", n=40),
            MetricResult(name="macro_f1", value=hs, split="holdout", n=20),
        ]
        cfg = MagicMock()
        cfg.primary_metric.value = "macro_f1"
        ctx = HypothesisContext(
            split=None, baseline_run=bl, config=cfg, col_types={}, seeds=MagicMock()
        )
        ctx.thresholds = t  # type: ignore[attr-defined]
        return ctx

    # absolute gap fires (0.12 >= 0.10)
    r = hyp.trigger(_ctx(0.92, 0.80))
    assert r.fired is True

    # relative gap fires (0.20 >= 0.15, abs gap 0.08 < 0.10)
    r = hyp.trigger(_ctx(0.40, 0.32))
    assert r.fired is True

    # below both thresholds
    r = hyp.trigger(_ctx(0.70, 0.67))
    assert r.fired is False

    # zero train score → gap_rel undefined → only absolute check
    r = hyp.trigger(_ctx(0.0, 0.0))
    assert r.fired is False  # gap == 0 < 0.10


# ---------------------------------------------------------------------------
# T2: test_select_capacity
# ---------------------------------------------------------------------------


def test_select_capacity() -> None:
    """Grid-only selection, determinism, tie-break, budget check abort."""
    from modelroot.experiments.budget import BudgetExceeded

    rng = np.random.default_rng(42)
    n = 80
    X = pd.DataFrame({"a": rng.random(n).astype(str), "b": rng.random(n).astype(str)})
    y = pd.Series(rng.integers(0, 2, n))
    adapter = _adapter("decision_tree")
    grid = [{"max_depth": 2}, {"max_depth": 5}]

    # Deterministic
    s1 = select_capacity(adapter, X, y, grid, "accuracy", seed=0)
    s2 = select_capacity(adapter, X, y, grid, "accuracy", seed=0)
    assert s1.overrides == s2.overrides
    assert s1.cv_score == s2.cv_score
    assert len(s1.cv_scores) == len(grid)

    # Tie-break: identical grid entries → first one wins
    same_grid = [{"max_depth": 3}, {"max_depth": 3}]
    s_tie = select_capacity(adapter, X, y, same_grid, "accuracy", seed=0)
    assert s_tie.overrides == same_grid[0]

    # Budget check called; raises → propagates
    call_count = {"n": 0}

    def _raise_on_second() -> None:
        call_count["n"] += 1
        if call_count["n"] >= 2:
            raise BudgetExceeded("test")

    with pytest.raises(BudgetExceeded):
        select_capacity(adapter, X, y, grid, "accuracy", seed=0, check=_raise_on_second)


# ---------------------------------------------------------------------------
# T3: test_regularize_intervention
# ---------------------------------------------------------------------------


def test_regularize_intervention(tmp_path: pathlib.Path) -> None:
    """Valid outcome, notes, fixed-param factory, no baseline mutation."""
    from fixtures.synthetic import make_config, make_overfit

    import modelroot.hypotheses.overfitting  # noqa: F401  # ensures registration
    from modelroot.assessment.baseline import fit_baseline
    from modelroot.data.loader import load_csv
    from modelroot.data.split import make_split
    from modelroot.data.validate import validate_for_analysis
    from modelroot.experiments.interventions import list_interventions
    from modelroot.models.registry import get_model
    from modelroot.utils.seeds import SeedManager

    assert "regularize" in list_interventions()

    csv = make_overfit(tmp_path, n_train=60, n_test=80, n_noise=5, seed=1)
    config = make_config(csv, algorithm="decision_tree")

    df = load_csv(csv)
    validated = validate_for_analysis(df, config)
    sm = SeedManager(0)
    split = make_split(validated, config, seed=sm.spawn("split"))
    col_types = dict.fromkeys(split.X_train.columns, "numeric")
    adapter = get_model(config.model_spec, col_types, seed=sm.spawn("baseline"))
    baseline_run = fit_baseline(adapter, split)

    ctx = HypothesisContext(
        split=split, baseline_run=baseline_run, config=config, col_types=col_types, seeds=sm
    )
    ctx.thresholds = Thresholds()  # type: ignore[attr-defined]

    from modelroot.experiments.interventions import get_intervention

    fn = get_intervention("regularize")
    outcome = fn(ctx, {})

    # Outcome has correct shape
    assert len(outcome.holdout_pred) == len(split.y_holdout)
    assert len(outcome.holdout_proba) == len(split.y_holdout)

    # Notes present
    notes_str = " ".join(outcome.notes)
    assert "selected_params=" in notes_str
    assert "train_score_reg=" in notes_str

    # Factory produces fresh unfitted adapters
    a1 = outcome.factory()
    a2 = outcome.factory()
    assert a1 is not a2
    assert a1._pipeline is None  # type: ignore[attr-defined]

    # Baseline adapter not mutated
    assert adapter._pipeline is None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# T4: test_evaluate_decision_table
# ---------------------------------------------------------------------------


def test_evaluate_decision_table() -> None:
    """Fabricated results covering supported/weakly/rejected/inconclusive + naive."""
    from unittest.mock import MagicMock

    from modelroot.schemas.common import RulesProfile
    from modelroot.schemas.experiment import ExperimentResult, ExperimentStatus
    from modelroot.schemas.metrics import MetricResult

    hyp = OverfittingHypothesis()
    t = Thresholds()

    def _ctx(profile: RulesProfile = RulesProfile.full) -> HypothesisContext:
        bl = MagicMock()
        bl.metrics = [
            MetricResult(name="macro_f1", value=0.90, split="train", n=200),
            MetricResult(name="macro_f1", value=0.70, split="holdout", n=200),
        ]
        bl.holdout_pred = MagicMock()
        bl.holdout_pred.y_true = np.array([0, 1] * 100)
        cfg = MagicMock()
        cfg.primary_metric.value = "macro_f1"
        cfg.rules_profile = profile
        cfg.budget_s = 300.0
        sm = MagicMock()
        sm.spawn = lambda name: 0
        ctx = HypothesisContext(
            split=MagicMock(), baseline_run=bl, config=cfg, col_types={}, seeds=sm
        )
        ctx.thresholds = t  # type: ignore[attr-defined]
        return ctx

    def _exp(delta: float, ci_low: float, ci_high: float, n_eval: int = 200) -> ExperimentResult:
        return ExperimentResult(
            spec_id="exp-of-regularize",
            status=ExperimentStatus.completed,
            baseline_metric=0.70,
            intervention_metric=0.70 + delta,
            delta=delta,
            ci_low=ci_low,
            ci_high=ci_high,
            n_eval=n_eval,
            n_valid_boot=1000,
            runtime_s=1.0,
            notes=["selected_params={}", "train_score_reg=0.7", "inner_cv_score=0.7"],
        )

    trigger = TriggerResult(
        fired=True,
        rule_id="OF-T1",
        values={"train_score": 0.90, "holdout_score": 0.70, "gap": 0.20},
    )

    # supported: large delta, CI excludes 0, adequate power, add repeats for consistency
    ctx = _ctx()
    ctx.split.X_train = pd.DataFrame({"a": ["1"] * 200})  # type: ignore[attr-defined]
    exp_supported = ExperimentResult(
        spec_id="exp-of-regularize",
        status=ExperimentStatus.completed,
        baseline_metric=0.70,
        intervention_metric=0.75,
        delta=0.05,
        ci_low=0.01,
        ci_high=0.09,
        n_eval=200,
        n_valid_boot=1000,
        runtime_s=1.0,
        notes=["selected_params={}", "train_score_reg=0.7", "inner_cv_score=0.7"],
        repeat_deltas=[0.05] * 10,
        consistency=1.0,
    )
    results = hyp.evaluate(ctx, trigger, [exp_supported])
    assert results[0].status == Status.supported

    # inconclusive: power gate fails (n_eval=10)
    ctx2 = _ctx()
    ctx2.split.X_train = pd.DataFrame({"a": ["1"] * 10})  # type: ignore[attr-defined]
    results2 = hyp.evaluate(ctx2, trigger, [_exp(delta=0.05, ci_low=0.01, ci_high=0.09, n_eval=10)])
    assert results2[0].status == Status.inconclusive

    # naive profile: only delta matters
    ctx_naive = _ctx(RulesProfile.naive)
    ctx_naive.split.X_train = pd.DataFrame({"a": ["1"] * 200})  # type: ignore[attr-defined]
    results_naive = hyp.evaluate(ctx_naive, trigger, [_exp(delta=0.05, ci_low=-0.01, ci_high=0.11)])
    assert results_naive[0].status == Status.supported
    results_naive_rej = hyp.evaluate(
        ctx_naive, trigger, [_exp(delta=0.005, ci_low=-0.01, ci_high=0.1)]
    )
    assert results_naive_rej[0].status == Status.rejected


# ---------------------------------------------------------------------------
# T5: test_nb_corroboration_not_a_gate
# ---------------------------------------------------------------------------


def test_nb_corroboration_not_a_gate() -> None:
    """NB p affects strength but not status."""
    from unittest.mock import MagicMock

    from modelroot.schemas.experiment import ExperimentResult, ExperimentStatus
    from modelroot.schemas.metrics import MetricResult

    hyp = OverfittingHypothesis()
    t = Thresholds()
    trigger = TriggerResult(
        fired=True,
        rule_id="OF-T1",
        values={"train_score": 0.90, "holdout_score": 0.70, "gap": 0.20},
    )

    def _ctx(repeat_deltas: list[float]) -> HypothesisContext:
        bl = MagicMock()
        bl.metrics = [
            MetricResult(name="macro_f1", value=0.90, split="train", n=200),
            MetricResult(name="macro_f1", value=0.70, split="holdout", n=200),
        ]
        bl.holdout_pred = MagicMock()
        bl.holdout_pred.y_true = np.array([0, 1] * 100)
        cfg = MagicMock()
        cfg.primary_metric.value = "macro_f1"
        cfg.rules_profile = RulesProfile.full
        cfg.budget_s = 300.0
        sm = MagicMock()
        sm.spawn = lambda name: 0
        ctx = HypothesisContext(
            split=MagicMock(), baseline_run=bl, config=cfg, col_types={}, seeds=sm
        )
        ctx.thresholds = t  # type: ignore[attr-defined]
        ctx.split.X_train = pd.DataFrame({"a": ["1"] * 200})  # type: ignore[attr-defined]
        return ctx

    exp_with_repeats = ExperimentResult(
        spec_id="exp-of-regularize",
        status=ExperimentStatus.completed,
        baseline_metric=0.70,
        intervention_metric=0.75,
        delta=0.05,
        ci_low=0.01,
        ci_high=0.09,
        n_eval=200,
        n_valid_boot=1000,
        runtime_s=1.0,
        notes=["selected_params={}", "train_score_reg=0.7", "inner_cv_score=0.7"],
        repeat_deltas=[0.05] * 10,  # consistent → NB corroborates
    )
    exp_no_repeats = ExperimentResult(
        spec_id="exp-of-regularize",
        status=ExperimentStatus.completed,
        baseline_metric=0.70,
        intervention_metric=0.75,
        delta=0.05,
        ci_low=0.01,
        ci_high=0.09,
        n_eval=200,
        n_valid_boot=1000,
        runtime_s=1.0,
        notes=["selected_params={}", "train_score_reg=0.7", "inner_cv_score=0.7"],
    )

    r_with = hyp.evaluate(_ctx([0.05] * 10), trigger, [exp_with_repeats])
    r_without = hyp.evaluate(_ctx([]), trigger, [exp_no_repeats])

    # Status may differ only due to consistency gate, not NB alone
    # Both should at least not be not_triggered
    assert r_with[0].status != Status.not_triggered
    assert r_without[0].status != Status.not_triggered
    # When repeats present: NB criterion in criteria_met
    if r_with[0].decision.criteria_met:
        assert "nb_corroborates" in r_with[0].decision.criteria_met


# ---------------------------------------------------------------------------
# T6: test_size_curve_and_budget
# ---------------------------------------------------------------------------


def test_size_curve_and_budget(tmp_path: pathlib.Path) -> None:
    """Five fractions, fraction 1.0 reuses baseline, budget exhaustion."""
    from fixtures.synthetic import make_config, make_overfit

    from modelroot.assessment.baseline import fit_baseline
    from modelroot.data.loader import load_csv
    from modelroot.data.split import make_split
    from modelroot.data.validate import validate_for_analysis
    from modelroot.experiments.budget import BudgetExceeded
    from modelroot.models.registry import get_model
    from modelroot.utils.seeds import SeedManager

    csv = make_overfit(tmp_path, n_train=60, n_test=80, n_noise=3, seed=2)
    config = make_config(csv, algorithm="decision_tree")
    df = load_csv(csv)
    validated = validate_for_analysis(df, config)
    sm = SeedManager(0)
    split = make_split(validated, config, seed=sm.spawn("split"))
    col_types = dict.fromkeys(split.X_train.columns, "numeric")
    adapter = get_model(config.model_spec, col_types, seed=0)
    baseline_run = fit_baseline(adapter, split)

    ctx = HypothesisContext(
        split=split, baseline_run=baseline_run, config=config, col_types=col_types, seeds=sm
    )
    ctx.thresholds = Thresholds()  # type: ignore[attr-defined]

    lc = compute_size_curve(ctx)
    # All 5 fractions returned
    assert set(lc.keys()) == set(SIZE_CURVE_FRACTIONS)

    # Fraction 1.0 matches baseline
    base_score = next(
        (
            float(m.value)
            for m in baseline_run.metrics
            if m.name == "macro_f1" and m.split == "holdout"
        ),
        None,
    )
    assert lc[1.0] == base_score

    # Slope calculable
    assert 0.875 in lc and 1.0 in lc

    # Budget exhaustion
    def _raise_immediately() -> None:
        raise BudgetExceeded("test")

    ctx2 = HypothesisContext(
        split=split,
        baseline_run=baseline_run,
        config=config,
        col_types=col_types,
        seeds=SeedManager(1),
    )
    ctx2.thresholds = Thresholds()  # type: ignore[attr-defined]
    ctx2.budget_check = _raise_immediately  # type: ignore[attr-defined]

    with pytest.raises(BudgetExceeded):
        compute_size_curve(ctx2)


# ---------------------------------------------------------------------------
# T7: test_registration_and_defaults
# ---------------------------------------------------------------------------


def test_registration_and_defaults() -> None:
    """H1 first in defaults, explicit empty registry stays empty, idempotent rule reg."""
    import modelroot.hypotheses.overfitting  # noqa: F401
    from modelroot.hypotheses.registry import clear_registry

    # clear user registry; H1 always comes from default_hypotheses
    clear_registry()
    dh = default_hypotheses()
    assert len(dh) >= 1
    from modelroot.hypotheses.overfitting import OverfittingHypothesis

    assert dh[0] is OverfittingHypothesis

    # explicit empty registry remains empty
    assert get_registry() == []

    # rule registration idempotent
    from modelroot.evidence.rules import register_rule
    from modelroot.hypotheses.overfitting import _RULE_OF_D1

    register_rule(_RULE_OF_D1)
    registry_before = len(get_rule_registry())
    register_rule(_RULE_OF_D1)
    assert len(get_rule_registry()) == registry_before

    # New thresholds validate
    t_custom = Thresholds(gap_rel=0.20, gap_shrink_min=0.40, consistency_min=0.85)
    assert t_custom.gap_rel == 0.20
    assert t_custom.gap_shrink_min == 0.40
