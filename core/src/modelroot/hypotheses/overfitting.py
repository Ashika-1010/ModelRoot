"""H1 — Overfitting hypothesis (M11)."""

from __future__ import annotations

import contextlib
import json
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from modelroot.assessment.metrics import UndefinedMetric, metric_fn, needs_proba
from modelroot.evidence.rules import (
    Criterion,
    Rule,
    decide_intervention_effect,
    evidence_strength,
    make_evidence_id,
    register_rule,
)
from modelroot.experiments.budget import BudgetExceeded
from modelroot.experiments.interventions import InterventionOutcome, intervention
from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.schemas.common import EvidenceKind, HypothesisType, Status
from modelroot.schemas.evidence import Evidence
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec
from modelroot.schemas.hypothesis import Decision, EffectResult, HypothesisResult, TriggerResult
from modelroot.stats.power import power_gate

SIZE_CURVE_FRACTIONS: tuple[float, ...] = (0.5, 0.625, 0.75, 0.875, 1.0)

_RULE_OF_D1 = Rule(
    id="OF-D1",
    text=(
        "Supported if Δ≥delta_min, CI excludes zero, adequately powered, "
        "gap shrinks ≥gap_shrink_min, consistency≥consistency_min. "
        "NB corroboration affects evidence strength only."
    ),
    criteria=[
        Criterion(id="delta_min", weight=1.0, description="Δ≥delta_min"),
        Criterion(id="ci_excludes_zero", weight=1.0, description="CI lower > 0"),
        Criterion(id="adequately_powered", weight=1.0, description="Power gate passed"),
        Criterion(id="gap_shrinks", weight=1.0, description="Gap shrinkage ≥ gap_shrink_min"),
        Criterion(id="consistent", weight=1.0, description="Consistency ≥ consistency_min"),
        Criterion(id="nb_corroborates", weight=1.0, description="NB p ≤ alpha"),
    ],
)
register_rule(_RULE_OF_D1)


@dataclass(frozen=True)
class CapacitySelection:
    """Result of inner-CV capacity selection."""

    overrides: dict[str, object]
    cv_score: float
    cv_scores: tuple[float, ...]


def _safe_float(v: float) -> float:
    f = float(v)
    return f if math.isfinite(f) else 0.0


def _get_metric(metrics: list[Any], name: str, split: str) -> float | None:
    for m in metrics:
        if m.name == name and m.split == split:
            return float(m.value)
    return None


def select_capacity(
    adapter: Any,
    X: pd.DataFrame,
    y: pd.Series,
    grid: list[dict[str, object]],
    metric: str,
    inner_cv: int = 3,
    *,
    seed: int,
    check: Callable[[], None] | None = None,
) -> CapacitySelection:
    """Select hyper-parameters by shuffled stratified inner CV."""
    if not grid:
        raise ValueError("Capacity grid must not be empty.")
    fn = metric_fn(metric)
    use_proba = needs_proba(metric)
    y_arr = np.asarray(y, dtype=int)
    kf = StratifiedKFold(n_splits=inner_cv, shuffle=True, random_state=seed)
    cv_scores: list[float] = []
    for overrides in grid:
        fold_scores: list[float] = []
        for train_idx, val_idx in kf.split(X, y_arr):
            if check is not None:
                check()
            X_tr = X.iloc[train_idx]
            y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else pd.Series(y_arr[train_idx])
            X_val = X.iloc[val_idx]
            y_val = y_arr[val_idx]
            candidate = adapter.clone_with(**overrides)
            candidate.fit(X_tr, y_tr)
            pred = np.asarray(candidate.predict(X_val))
            proba = (
                np.asarray(candidate.predict_proba(X_val))[:, 1]
                if use_proba
                else np.zeros(len(y_val))
            )
            fold_scores.append(float(fn(y_val, pred, proba)))
        cv_scores.append(float(np.mean(fold_scores)))
    best_idx = int(np.argmax(cv_scores))
    return CapacitySelection(
        overrides=dict(grid[best_idx]),
        cv_score=cv_scores[best_idx],
        cv_scores=tuple(cv_scores),
    )


@intervention("regularize")
def _regularize(ctx: HypothesisContext, params: dict[str, object]) -> InterventionOutcome:
    """Select regularised config and fit on training workspace only."""
    adapter = ctx.baseline_run.adapter
    split = ctx.split
    metric = ctx.config.primary_metric.value
    seed = ctx.seeds.spawn("overfitting:regularize")
    grid = adapter.capacity_grid()
    if not grid:
        raise ValueError("Adapter capacity grid is empty.")
    check_fn = ctx.budget_check
    sel = select_capacity(
        adapter=adapter,
        X=split.X_train,
        y=split.y_train,
        grid=grid,
        metric=metric,
        inner_cv=3,
        seed=seed,
        check=check_fn,
    )
    reg_adapter = adapter.clone_with(**sel.overrides)
    reg_adapter.fit(split.X_train, split.y_train)
    fn = metric_fn(metric)
    use_proba = needs_proba(metric)
    train_pred = np.asarray(reg_adapter.predict(split.X_train))
    train_proba = (
        np.asarray(reg_adapter.predict_proba(split.X_train))[:, 1]
        if use_proba
        else np.zeros(len(split.y_train))
    )
    train_s = _safe_float(fn(np.asarray(split.y_train, dtype=int), train_pred, train_proba))
    holdout_pred_arr = np.asarray(reg_adapter.predict(split.X_holdout))
    holdout_proba_arr = np.asarray(reg_adapter.predict_proba(split.X_holdout))[:, 1]
    notes = [
        "selected_params=" + json.dumps(sel.overrides, sort_keys=True),
        "train_score_reg=" + repr(train_s),
        "inner_cv_score=" + repr(sel.cv_score),
    ]
    fixed_overrides = dict(sel.overrides)

    def _factory() -> object:
        return adapter.clone_with(**fixed_overrides)

    return InterventionOutcome(
        holdout_pred=holdout_pred_arr,
        holdout_proba=holdout_proba_arr,
        fitted_adapter=reg_adapter,
        factory=_factory,
        notes=notes,
    )


def compute_size_curve(ctx: HypothesisContext) -> dict[float, float]:
    """Compute five-point learning curve. Returns fraction → holdout score."""
    from modelroot.models.registry import get_model

    split = ctx.split
    config = ctx.config
    metric = config.primary_metric.value
    fn = metric_fn(metric)
    use_proba = needs_proba(metric)
    y_holdout = np.asarray(split.y_holdout, dtype=int)
    col_types = ctx.col_types

    score_1: float = 0.0
    for m in ctx.baseline_run.metrics:
        if m.name == metric and m.split == "holdout":
            score_1 = float(m.value)
            break
    results: dict[float, float] = {1.0: score_1}

    X_train = split.X_train
    y_train_arr = np.asarray(split.y_train, dtype=int)
    n_train = len(y_train_arr)

    for frac in SIZE_CURVE_FRACTIONS[:-1]:
        seed = ctx.seeds.spawn(f"overfitting:size_curve:{frac}")
        if ctx.budget_check is not None:
            ctx.budget_check()  # may raise BudgetExceeded
        n_sub = max(2, round(n_train * frac))
        rng = np.random.default_rng(seed)
        unique, counts = np.unique(y_train_arr, return_counts=True)
        sub_idx: list[int] = []
        for cls_val, cnt in zip(unique, counts, strict=False):
            cls_idx = np.where(y_train_arr == cls_val)[0]
            n_cls = max(1, round(n_sub * cnt / n_train))
            chosen = rng.choice(cls_idx, size=min(n_cls, len(cls_idx)), replace=False)
            sub_idx.extend(chosen.tolist())
        sub_arr = np.array(sub_idx[:n_sub], dtype=int)
        X_sub = X_train.iloc[sub_arr]
        y_sub = pd.Series(y_train_arr[sub_arr])
        try:
            ada = get_model(config.model_spec, col_types, seed=seed)
            ada.fit(X_sub, y_sub)
            pred = np.asarray(ada.predict(split.X_holdout))
            proba = (
                np.asarray(ada.predict_proba(split.X_holdout))[:, 1]
                if use_proba
                else np.zeros(len(y_holdout))
            )
            results[frac] = _safe_float(fn(y_holdout, pred, proba))
        except (ValueError, UndefinedMetric):
            continue
    return results


class OverfittingHypothesis(Hypothesis):
    """H1 — Overfitting (M11)."""

    type = HypothesisType.overfitting

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        metric = ctx.config.primary_metric.value
        ts = _get_metric(ctx.baseline_run.metrics, metric, "train")
        hs = _get_metric(ctx.baseline_run.metrics, metric, "holdout")
        if ts is None or hs is None:
            return Applicability(applicable=False, reason="Train/holdout metric unavailable.")
        return Applicability(applicable=True)

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        from modelroot.config.thresholds import Thresholds

        thresholds: Thresholds = ctx.thresholds if ctx.thresholds is not None else Thresholds()
        metric = ctx.config.primary_metric.value
        ts = _get_metric(ctx.baseline_run.metrics, metric, "train") or 0.0
        hs = _get_metric(ctx.baseline_run.metrics, metric, "holdout") or 0.0
        gap = ts - hs
        gap_rel: float | None = (gap / ts) if ts > 0 else None
        fired = gap >= thresholds.gap or (gap_rel is not None and gap_rel >= thresholds.gap_rel)
        vals: dict[str, float | str | bool | int | None] = {
            "train_score": _safe_float(ts),
            "holdout_score": _safe_float(hs),
            "gap": _safe_float(gap),
        }
        if gap_rel is not None:
            vals["gap_rel"] = _safe_float(gap_rel)
        return TriggerResult(fired=fired, rule_id="OF-T1", values=vals)

    def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list[ExperimentSpec]:
        from modelroot.config.thresholds import Thresholds

        thresholds: Thresholds = ctx.thresholds if ctx.thresholds is not None else Thresholds()
        seed = ctx.seeds.spawn("overfitting:regularize")
        return [
            ExperimentSpec(
                id="exp-of-regularize",
                hypothesis_id="h1-overfitting",
                name="regularize",
                intervention="regularize",
                params={},
                metric=ctx.config.primary_metric.value,
                eval_on=["holdout"],
                n_boot=thresholds.n_boot,
                repeats=thresholds.repeats,
                seed=seed,
                budget_s=ctx.config.budget_s,
            )
        ]

    def evaluate(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
        results: list[Any],
    ) -> list[HypothesisResult]:
        from modelroot.config.thresholds import Thresholds

        thresholds: Thresholds = ctx.thresholds if ctx.thresholds is not None else Thresholds()
        metric = ctx.config.primary_metric.value
        profile = ctx.config.rules_profile
        ts = _get_metric(ctx.baseline_run.metrics, metric, "train") or 0.0
        hs = _get_metric(ctx.baseline_run.metrics, metric, "holdout") or 0.0
        gap_base = ts - hs

        caveats = [
            "The gap may also arise from label noise or train/holdout mismatch.",
            "The regularization grid is limited; a better configuration may exist.",
        ]
        ev_items: list[Evidence] = []
        diag_ev_ids: list[str] = []

        def _add_ev(name: str, val: float, method: str) -> str:
            eid = make_evidence_id(name, val, method, None)
            ev_items.append(
                Evidence(
                    id=eid,
                    kind=EvidenceKind.diagnostic,
                    name=name,
                    value=_safe_float(val),
                    method=method,
                    hint=f"{name}={_safe_float(val):.4f}",
                )
            )
            return eid

        diag_ev_ids.append(_add_ev("train_score", ts, "metric_computation"))
        diag_ev_ids.append(_add_ev("holdout_score", hs, "metric_computation"))
        diag_ev_ids.append(_add_ev("gap", gap_base, "metric_difference"))
        grv = trigger.values.get("gap_rel")
        if isinstance(grv, int | float):
            diag_ev_ids.append(_add_ev("gap_rel", float(grv), "relative_gap"))

        # Learning curve (best-effort)
        lc_slope: float | None = None
        try:
            lc = compute_size_curve(ctx)
            if 0.875 in lc and 1.0 in lc:
                lc_slope = (lc[1.0] - lc[0.875]) / 0.125
                diag_ev_ids.append(_add_ev("lc_slope", _safe_float(lc_slope), "learning_curve"))
        except BudgetExceeded:
            caveats.append("Learning-curve evidence was unavailable: time budget exhausted.")
        except (ValueError, UndefinedMetric):
            caveats.append(
                "Learning-curve evidence was unavailable: dataset too small for subsampling."
            )

        if not results:
            status = Status.not_triggered if not trigger.fired else Status.inconclusive
            criteria_met: dict[str, bool] = {}
            strength = 0.0
            exp_ids: list[str] = []
            effect: EffectResult | None = None
            recommended = _recommendations(status, lc_slope)
            return [
                HypothesisResult(
                    id="h1-overfitting",
                    type=HypothesisType.overfitting,
                    status=status,
                    evidence_strength=strength,
                    trigger=trigger,
                    diagnostic_evidence_ids=diag_ev_ids,
                    experiment_ids=exp_ids,
                    effect=effect,
                    decision=Decision(
                        rule_id="OF-D1",
                        rule_text=_RULE_OF_D1.text,
                        rules_profile=profile,
                        criteria_met=criteria_met,
                    ),
                    caveats=caveats,
                    recommended_actions=recommended,
                )
            ]

        exp: ExperimentResult = results[0]
        exp_ids = [exp.spec_id]

        # Parse notes
        selected_params: dict[str, object] = {}
        train_score_reg: float = hs
        for note in getattr(exp, "notes", []):
            if note.startswith("selected_params="):
                with contextlib.suppress(json.JSONDecodeError):
                    selected_params = json.loads(note[len("selected_params=") :])
            elif note.startswith("train_score_reg="):
                with contextlib.suppress(Exception):
                    train_score_reg = float(eval(note[len("train_score_reg=") :]))

        intervention_holdout = exp.intervention_metric
        gap_reg = train_score_reg - intervention_holdout
        gap_shrinkage = 1.0 if abs(gap_base) < 1e-12 else 1.0 - gap_reg / gap_base

        # Power gate
        n_eval = exp.n_eval
        n_pos = int(np.sum(ctx.baseline_run.holdout_pred.y_true == 1))
        ci_width = exp.ci_high - exp.ci_low
        gate = power_gate(
            n_holdout=n_eval,
            n_pos=n_pos,
            ci_width=ci_width,
            holdout_min_n=thresholds.holdout_min_n,
            holdout_min_pos=thresholds.holdout_min_pos,
            ci_width_max=thresholds.ci_width_max,
        )

        # Repeated subsampling consistency + NB (consumed from ExperimentResult)
        rep_deltas = list(getattr(exp, "repeat_deltas", []))
        cons_val: float = (
            float(exp.consistency)
            if exp.consistency is not None
            else (float(sum(d > 0 for d in rep_deltas) / len(rep_deltas)) if rep_deltas else 0.0)
        )
        nb_p: float = float(exp.nb.p) if exp.nb is not None else 1.0

        # Extra criteria (full profile)
        extra = {
            "gap_shrinks": gap_shrinkage >= thresholds.gap_shrink_min,
            "consistent": cons_val >= thresholds.consistency_min,
        }
        status, criteria_met = decide_intervention_effect(
            delta=exp.delta,
            ci_low=exp.ci_low,
            ci_high=exp.ci_high,
            gate_ok=gate.ok,
            extra_criteria=extra,
            thresholds=thresholds,
            profile=profile,
        )
        # NB corroboration: strength only (not a gate)
        criteria_met["nb_corroborates"] = nb_p <= thresholds.alpha
        strength = evidence_strength(_RULE_OF_D1, criteria_met)

        diag_ev_ids.append(_add_ev("gap_shrinkage", _safe_float(gap_shrinkage), "gap_shrinkage"))
        diag_ev_ids.append(_add_ev("consistency", _safe_float(cons_val), "consistency"))

        effect = EffectResult(
            metric=metric,
            baseline=_safe_float(exp.baseline_metric),
            intervention=_safe_float(intervention_holdout),
            delta=_safe_float(exp.delta),
            ci_low=_safe_float(exp.ci_low),
            ci_high=_safe_float(exp.ci_high),
            n=n_eval,
        )

        recommended = _recommendations(status, lc_slope, selected_params)

        return [
            HypothesisResult(
                id="h1-overfitting",
                type=HypothesisType.overfitting,
                status=status,
                evidence_strength=_safe_float(strength),
                trigger=trigger,
                diagnostic_evidence_ids=diag_ev_ids,
                experiment_ids=exp_ids,
                effect=effect,
                decision=Decision(
                    rule_id="OF-D1",
                    rule_text=_RULE_OF_D1.text,
                    rules_profile=profile,
                    criteria_met=criteria_met,
                ),
                caveats=caveats,
                recommended_actions=recommended,
            )
        ]


def _recommendations(
    status: Status,
    lc_slope: float | None,
    selected_params: dict[str, object] | None = None,
) -> list[str]:
    actions: list[str] = []
    if status in (Status.supported, Status.weakly_supported):
        if selected_params:
            actions.append(
                f"Consider using the regularised configuration: "
                f"{json.dumps(selected_params, sort_keys=True)}"
            )
        else:
            actions.append("Reduce model capacity using the selected regularised configuration.")
        if lc_slope is not None and lc_slope > 0:
            actions.append("The learning curve suggests collecting more training data may help.")
    elif status == Status.rejected:
        actions.append("Check for label noise and train/holdout distribution mismatch.")
    else:
        actions.append("Collect more holdout data or increase the number of repeats.")
    return actions
