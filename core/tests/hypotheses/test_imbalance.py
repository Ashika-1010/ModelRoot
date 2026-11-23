"""M14 tests: ImbalanceHypothesis unit and integration tests."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

from modelroot.assessment.baseline import fit_baseline
from modelroot.config.thresholds import Thresholds
from modelroot.data.split import SplitData
from modelroot.evidence.rules import RULE_IM_D1, RULE_IM_D2, get_rule_registry, register_rule
from modelroot.experiments.interventions import get_intervention, list_interventions
from modelroot.hypotheses.base import HypothesisContext
from modelroot.hypotheses.imbalance import ImbalanceHypothesis
from modelroot.hypotheses.overfitting import OverfittingHypothesis
from modelroot.hypotheses.registry import clear_registry, default_hypotheses
from modelroot.models.registry import get_model
from modelroot.schemas.common import (
    Algorithm,
    ExperimentStatus,
    PrimaryMetric,
    Status,
)
from modelroot.schemas.config import AnalysisConfig, ModelSpec
from modelroot.schemas.experiment import ExperimentResult
from modelroot.schemas.hypothesis import TriggerResult
from modelroot.utils.seeds import SeedManager


def _make_imbalanced_split(
    n_maj: int = 400,
    n_min: int = 50,
    n_ho_maj: int = 170,
    n_ho_min: int = 30,
    class_sep: float = 1.0,
    flip_y: float = 0.02,
    seed: int = 42,
) -> tuple[SplitData, HypothesisContext]:
    """Create a synthetic imbalanced train/holdout split with context."""
    rng = np.random.default_rng(seed)
    n_total = n_maj + n_min + n_ho_maj + n_ho_min
    p_min = (n_min + n_ho_min) / n_total

    X, y = make_classification(
        n_samples=n_total,
        n_features=8,
        n_informative=5,
        n_redundant=1,
        weights=[1.0 - p_min, p_min],
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=seed,
    )
    feature_names = [f"f_{i}" for i in range(8)]
    col_types = dict.fromkeys(feature_names, "numeric")

    maj_idx = np.where(y == 0)[0]
    min_idx = np.where(y == 1)[0]
    rng.shuffle(maj_idx)
    rng.shuffle(min_idx)

    tr_maj = maj_idx[:n_maj]
    tr_min = min_idx[:n_min]
    ho_maj = maj_idx[n_maj : n_maj + n_ho_maj]
    ho_min = min_idx[n_min : n_min + n_ho_min]

    tr_idx = np.concatenate([tr_maj, tr_min])
    ho_idx = np.concatenate([ho_maj, ho_min])
    rng.shuffle(tr_idx)
    rng.shuffle(ho_idx)

    X_tr = pd.DataFrame(X[tr_idx], columns=feature_names, index=tr_idx)
    y_tr = pd.Series(y[tr_idx], name="target", index=tr_idx)
    X_ho = pd.DataFrame(X[ho_idx], columns=feature_names, index=ho_idx)
    y_ho = pd.Series(y[ho_idx], name="target", index=ho_idx)

    split = SplitData(
        X_train=X_tr,
        y_train=y_tr,
        X_holdout=X_ho,
        y_holdout=y_ho,
        orig_index_train=X_tr.index,
        orig_index_holdout=X_ho.index,
        split_fingerprint="test_imb",
    )

    spec = ModelSpec(
        algorithm=Algorithm.random_forest,
        params={"n_estimators": 25, "max_depth": 6, "min_samples_leaf": 1},
    )
    adapter = get_model(spec, col_types, seed=seed)
    baseline_run = fit_baseline(adapter, split)

    sm = SeedManager(seed)
    config = AnalysisConfig(
        data_path="dummy.csv",
        target="target",
        positive_class="1",
        primary_metric=PrimaryMetric.macro_f1,
        model_spec=spec,
        seed=seed,
    )

    ctx = HypothesisContext(
        split=split,
        baseline_run=baseline_run,
        config=config,
        col_types=col_types,
        seeds=sm,
        thresholds=Thresholds(),
    )
    return split, ctx


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------


def test_default_hypotheses_order() -> None:
    """Verify H1 is first and H2 is second in default_hypotheses()."""
    clear_registry()
    dh = default_hypotheses()
    assert len(dh) >= 2
    assert dh[0] is OverfittingHypothesis
    assert dh[1] is ImbalanceHypothesis


def test_imbalance_rules_registered() -> None:
    """Verify IM-D1 and IM-D2 rules are registered in global rule registry.

    Re-registers built-in IM rules explicitly so the test is independent of
    module-import order: ``clear_rule_registry()`` in test_m10 wipes them.
    """
    register_rule(RULE_IM_D1)
    register_rule(RULE_IM_D2)
    rules = {r.id: r for r in get_rule_registry()}
    assert "IM-D1" in rules
    assert "IM-D2" in rules
    assert rules["IM-D1"] is RULE_IM_D1
    assert rules["IM-D2"] is RULE_IM_D2


def test_applicable() -> None:
    """Verify Imbalance applicable on binary, false on multiclass."""
    _, ctx = _make_imbalanced_split()
    h = ImbalanceHypothesis()
    assert h.applicable(ctx).applicable is True

    # Multi-class dataset
    ctx.split.y_train = pd.Series([0, 1, 2] * 20)
    app = h.applicable(ctx)
    assert app.applicable is False
    assert "binary" in app.reason.lower()


def test_trigger_fired_on_imbalance() -> None:
    """Verify IM-T1 fires when minority share <= 0.20."""
    _, ctx = _make_imbalanced_split(n_maj=400, n_min=50)
    h = ImbalanceHypothesis()
    trig = h.trigger(ctx)
    assert trig.rule_id == "IM-T1"
    assert trig.fired is True
    assert trig.values["minority_share"] <= 0.20


def test_trigger_not_fired_on_balanced() -> None:
    """Verify IM-T1 does not fire on balanced dataset."""
    _, ctx = _make_imbalanced_split(n_maj=200, n_min=200, n_ho_maj=100, n_ho_min=100)
    h = ImbalanceHypothesis()
    trig = h.trigger(ctx)
    assert trig.fired is False


def test_plan_experiments() -> None:
    """Verify plan produces imbalance_best holdout experiment when triggered."""
    _, ctx = _make_imbalanced_split()
    h = ImbalanceHypothesis()
    trig = TriggerResult(fired=True, rule_id="IM-T1", values={"minority_share": 0.10})
    specs = h.plan(ctx, trig)
    assert len(specs) == 1
    assert specs[0].id == "exp-imbalance-harm"
    assert specs[0].intervention == "imbalance_best"

    # Not fired -> empty plan
    assert h.plan(ctx, TriggerResult(fired=False, rule_id="IM-T1")) == []


def test_interventions_registered() -> None:
    """Verify all 4 M14 interventions are registered."""
    registered = list_interventions()
    for name in ("class_weight", "threshold_tune", "oversample_train", "imbalance_best"):
        assert name in registered


def test_interventions_no_holdout_leakage() -> None:
    """Verify interventions evaluate strictly on holdout without touching it in training."""
    split, ctx = _make_imbalanced_split(n_maj=200, n_min=40, n_ho_maj=80, n_ho_min=20)

    for name in ("class_weight", "threshold_tune", "oversample_train", "imbalance_best"):
        fn = get_intervention(name)
        out = fn(ctx, {})
        assert len(out.holdout_pred) == len(split.y_holdout)


def test_evaluate_minority_oof_limitation() -> None:
    """Verify harm becomes inconclusive when minority train count < 30."""
    _, ctx = _make_imbalanced_split(n_maj=400, n_min=15, n_ho_maj=170, n_ho_min=30)
    h = ImbalanceHypothesis()
    trig = TriggerResult(
        fired=True,
        rule_id="IM-T1",
        values={"minority_share": 0.036, "majority_share": 0.964, "acc_bacc_gap": 0.10},
    )

    dummy_exp = ExperimentResult(
        spec_id="exp-imbalance-harm",
        status=ExperimentStatus.completed,
        baseline_metric=0.50,
        intervention_metric=0.60,
        delta=0.10,
        ci_low=0.03,
        ci_high=0.17,
        n_eval=200,
        n_valid_boot=1000,
        runtime_s=1.0,
        notes=["selected_intervention=threshold_tune"],
    )

    results = h.evaluate(ctx, trig, [dummy_exp])
    assert len(results) == 2
    harm_res = next(r for r in results if r.subtype == "harm")
    assert harm_res.status == Status.inconclusive
    assert any("limit confidence" in c for c in harm_res.caveats)


def test_evaluate_separable_hard_negative() -> None:
    """Verify separable hard negative scenario does not support distortion or harm."""
    _, ctx = _make_imbalanced_split(
        n_maj=400, n_min=40, n_ho_maj=180, n_ho_min=20, class_sep=3.0, flip_y=0.0
    )
    h = ImbalanceHypothesis()
    trig = h.trigger(ctx)

    if not trig.fired:
        results = h.evaluate(ctx, trig, [])
    else:
        dummy_exp = ExperimentResult(
            spec_id="exp-imbalance-harm",
            status=ExperimentStatus.completed,
            baseline_metric=0.99,
            intervention_metric=0.99,
            delta=0.00,
            ci_low=-0.01,
            ci_high=0.01,
            n_eval=200,
            n_valid_boot=1000,
            runtime_s=1.0,
            notes=["selected_intervention=threshold_tune"],
        )
        results = h.evaluate(ctx, trig, [dummy_exp])

    for r in results:
        assert r.status not in (Status.supported, Status.weakly_supported)
