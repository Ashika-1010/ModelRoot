"""Subgroup Underperformance Hypothesis (H4: M18).

Tests whether specific feature-defined subgroups suffer significantly higher error rates
than their complement, using training-only discovery, holdout confirmation, and
Benjamini-Yekutieli (BY) FDR control over the screened testable candidate family
with an exact unconditional margin test.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from modelroot.config.thresholds import Thresholds
from modelroot.evidence.rules import Criterion, Rule
from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.hypotheses.slices import Slice, enumerate_slices, exact_margin_test, get_slice_mask
from modelroot.schemas.common import HypothesisType, RulesProfile, Severity, Status
from modelroot.schemas.config import JsonScalar
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec
from modelroot.schemas.hypothesis import Decision, EffectResult, HypothesisResult, TriggerResult
from modelroot.stats.multiple_testing import benjamini_hochberg, benjamini_yekutieli
from modelroot.stats.permutation import permutation_test_two_sample

_RULE_SG_D1 = Rule(
    id="SG-D1",
    text=(
        "Supported if at least one candidate subgroup slice is confirmed on the holdout set "
        "under Benjamini-Yekutieli FDR control (q <= slice_fdr_q) using the exact unconditional "
        "margin test. Subtype is representation_related if reweighting improves slice error by "
        ">= delta_min; else intrinsic_or_unresolved. "
        "Inconclusive if candidates exist but n < slice_min_n. "
        "Rejected if tested but none confirmed."
    ),
    criteria=[
        Criterion(
            id="slice_confirmed",
            weight=2.0,
            description="At least one slice confirmed under BY FDR (q <= slice_fdr_q)",
        ),
        Criterion(
            id="testable_sample_size",
            weight=1.0,
            description="Holdout slice size n >= slice_min_n",
        ),
        Criterion(
            id="reweighting_effective",
            weight=1.0,
            description="Reweighting improves slice error by >= delta_min",
        ),
    ],
)


def _get_split(
    ctx: HypothesisContext,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Extract train and holdout splits across real SplitData and test double representations."""
    split = ctx.split
    if hasattr(split, "X_train"):
        return split.X_train, split.y_train, split.X_holdout, split.y_holdout
    if hasattr(split, "train"):
        return split.train.X, split.train.y, split.holdout.X, split.holdout.y
    raise ValueError("Unknown split structure in HypothesisContext")


class SubgroupHypothesis(Hypothesis):
    """H4: Subgroup Underperformance hypothesis.

    Discovers candidate underperforming slices on training OOF predictions and
    confirms them on the independent holdout set with exact unconditional margin testing
    and BY FDR control.
    """

    type: HypothesisType = HypothesisType.subgroup

    def __init__(self) -> None:
        super().__init__()
        # In-memory candidate cache tied to context id
        self._cached_candidates: list[Slice] | None = None
        self._cached_ctx_id: int | None = None

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        """Check whether H4 is applicable to this dataset."""
        if ctx.split is None:
            return Applicability(applicable=False, reason="Missing split")

        try:
            X_tr, _, X_ho, _ = _get_split(ctx)
        except Exception:
            return Applicability(applicable=False, reason="Invalid split structure")

        if X_tr is None or X_ho is None or X_tr.empty or len(X_tr.columns) == 0:
            return Applicability(applicable=False, reason="No feature columns available")

        if ctx.baseline_run is None:
            return Applicability(applicable=False, reason="Baseline model run not available")

        return Applicability(applicable=True)

    def _get_oof_predictions(self, ctx: HypothesisContext) -> np.ndarray:
        """Obtain out-of-fold predictions on the training workspace."""
        # 1. Reuse existing baseline OOF predictions if available
        if hasattr(ctx.baseline_run, "train_pred") and ctx.baseline_run.train_pred is not None:
            tr_pred = ctx.baseline_run.train_pred
            return np.asarray(getattr(tr_pred, "y_pred", tr_pred), dtype=int)
        if (
            hasattr(ctx.baseline_run, "oof_predictions")
            and ctx.baseline_run.oof_predictions is not None
        ):
            return np.asarray(ctx.baseline_run.oof_predictions, dtype=int)

        # 2. Otherwise compute 5-fold cross_val_predict on train split
        X_tr, y_tr, _, _ = _get_split(ctx)
        n_samples = len(y_tr)
        oof_preds = np.zeros(n_samples, dtype=int)

        from modelroot.models.registry import get_model

        skf = StratifiedKFold(
            n_splits=5, shuffle=True, random_state=ctx.seed_for("subgroup", "oof")
        )

        for train_idx, val_idx in skf.split(X_tr, y_tr):
            adapter = get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
            adapter.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
            oof_preds[val_idx] = adapter.predict(X_tr.iloc[val_idx])

        return oof_preds

    def _discover_candidates(self, ctx: HypothesisContext) -> list[Slice]:
        """Perform training-only discovery screening for underperforming slices."""
        thresholds = ctx.thresholds or Thresholds()
        X_tr, y_tr_s, _, _ = _get_split(ctx)
        y_tr = y_tr_s.to_numpy()

        oof_preds = self._get_oof_predictions(ctx)
        # 0 = correct, 1 = error
        errors_tr = (oof_preds != y_tr).astype(np.float64)

        all_slices = enumerate_slices(
            X_tr,
            ctx.col_types,
            min_n=thresholds.slice_min_n,
        )

        if not all_slices:
            return []

        scored_slices: list[Slice] = []
        for sl in all_slices:
            mask = get_slice_mask(X_tr, sl)
            n_slice = int(np.sum(mask))
            n_comp = len(X_tr) - n_slice

            if n_slice < thresholds.slice_min_n or n_comp < thresholds.slice_min_n:
                continue

            err_slice = float(np.mean(errors_tr[mask]))
            err_comp = float(np.mean(errors_tr[~mask]))
            gap = err_slice - err_comp

            sl.n_train = n_slice
            sl.error_train = err_slice
            sl.complement_error_train = err_comp
            sl.gap_train = gap

            # Compute discovery screening p-value using fast permutation test
            perm_res = permutation_test_two_sample(
                errors_tr[mask],
                errors_tr[~mask],
                alternative="greater",
                n_permutations=200,
                seed=ctx.seed_for("subgroup", "disc", sl.name),
            )
            sl.p_train = perm_res.p_value
            scored_slices.append(sl)

        if not scored_slices:
            return []

        # Check protocol ablation
        protocol = RulesProfile.full
        if (
            ctx.config
            and hasattr(ctx.config, "ablation")
            and hasattr(ctx.config.ablation, "subgroup_protocol")
        ):
            protocol = ctx.config.ablation.subgroup_protocol

        if protocol == RulesProfile.naive:
            # Naive ablation: rank purely by gap without FDR screening
            scored_slices.sort(key=lambda s: s.gap_train, reverse=True)
            candidates = [s for s in scored_slices if s.gap_train >= thresholds.slice_gap]
            return candidates[: thresholds.slice_max_candidates]

        # Full protocol: BH screening on discovery at q <= 0.20
        p_vals = [s.p_train if s.p_train is not None else 1.0 for s in scored_slices]
        q_vals = benjamini_hochberg(p_vals)

        screened: list[Slice] = []
        for sl, q in zip(scored_slices, q_vals, strict=True):
            sl.q_train = q
            if q <= 0.20 and sl.gap_train >= thresholds.slice_gap:
                screened.append(sl)

        screened.sort(key=lambda s: s.gap_train, reverse=True)
        return screened[: thresholds.slice_max_candidates]

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        """Run training-only discovery screen and cache candidate slices."""
        # Reset cache on every trigger call to ensure fresh state per run
        self._cached_candidates = None
        self._cached_ctx_id = None

        candidates = self._discover_candidates(ctx)

        if not candidates:
            return TriggerResult(
                fired=False,
                rule_id="SG-T1",
                values={
                    "fired": False,
                    "n_candidates": 0,
                    "max_gap_train": 0.0,
                },
            )

        # Cache candidates for plan and evaluate
        self._cached_candidates = candidates
        self._cached_ctx_id = id(ctx)

        top_slice = candidates[0]
        return TriggerResult(
            fired=True,
            rule_id="SG-T1",
            values={
                "fired": True,
                "n_candidates": len(candidates),
                "max_gap_train": float(top_slice.gap_train),
                "top_feature": str(top_slice.feature),
                "top_slice": str(top_slice.name),
            },
        )

    def plan(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
    ) -> list[ExperimentSpec]:
        """Plan mechanism experiments (reweight_group) for candidate slices."""
        if not trigger.fired:
            return []

        # Retrieve candidates from cache, or recompute if context changed
        candidates = self._cached_candidates
        if candidates is None or self._cached_ctx_id != id(ctx):
            candidates = self._discover_candidates(ctx)

        if not candidates:
            return []

        top_slice = candidates[0]
        specs: list[ExperimentSpec] = []

        if ctx.config and getattr(ctx.config, "run_experiments", False):
            metric_str = (
                ctx.config.primary_metric.value
                if hasattr(ctx.config, "primary_metric")
                and hasattr(ctx.config.primary_metric, "value")
                else "accuracy"
            )
            seed_val = (
                ctx.seed_for("experiment", "subgroup_reweight")
                if hasattr(ctx, "seed_for")
                else 2026
            )
            params: dict[str, JsonScalar] = {
                "feature": top_slice.feature,
                "value": str(top_slice.value) if top_slice.value is not None else "",
                "is_numeric_bin": top_slice.is_numeric_bin,
                "weight_cap": 5.0,
            }
            if top_slice.is_numeric_bin and top_slice.bin_edges:
                params["bin_low"] = float(top_slice.bin_edges[0])
                params["bin_high"] = float(top_slice.bin_edges[1])

            specs.append(
                ExperimentSpec(
                    id="E-SG-2",
                    hypothesis_id="h4-subgroup",
                    name=f"Reweight underperforming slice '{top_slice.name}'",
                    intervention="reweight_group",
                    params=params,
                    metric=metric_str,
                    seed=seed_val,
                    budget_s=30.0,
                )
            )

        return specs

    def evaluate(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
        results: list[ExperimentResult],
    ) -> list[HypothesisResult]:
        """Confirm candidate slices on independent holdout.

        Uses certified exact margin test and BY FDR control.
        """
        thresholds = ctx.thresholds or Thresholds()
        profile = (
            ctx.config.rules_profile
            if ctx.config and hasattr(ctx.config, "rules_profile")
            else RulesProfile.full
        )

        if not trigger.fired:
            return [
                HypothesisResult(
                    id="h4-subgroup",
                    type=HypothesisType.subgroup,
                    subtype=None,
                    status=Status.not_triggered,
                    evidence_strength=0.0,
                    trigger=trigger,
                    diagnostic_evidence_ids=[],
                    experiment_ids=[],
                    decision=Decision(
                        rule_id="SG-D1",
                        rule_text=_RULE_SG_D1.text,
                        rules_profile=profile,
                        criteria_met={"slice_confirmed": False, "testable_sample_size": False},
                    ),
                    caveats=["No underperforming slices identified during training discovery."],
                    recommended_actions=["Monitor slice-level performance during production."],
                )
            ]

        # Retrieve candidates from cache
        candidates = self._cached_candidates
        if candidates is None or self._cached_ctx_id != id(ctx):
            candidates = self._discover_candidates(ctx)

        if not candidates:
            return [
                HypothesisResult(
                    id="h4-subgroup",
                    type=HypothesisType.subgroup,
                    subtype=None,
                    status=Status.not_triggered,
                    evidence_strength=0.0,
                    trigger=trigger,
                    diagnostic_evidence_ids=[],
                    experiment_ids=[],
                    decision=Decision(
                        rule_id="SG-D1",
                        rule_text=_RULE_SG_D1.text,
                        rules_profile=profile,
                        criteria_met={"slice_confirmed": False, "testable_sample_size": False},
                    ),
                    caveats=["Candidate slices could not be reconstructed."],
                )
            ]

        # Holdout confirmation evaluation
        _, _, X_ho, y_ho_s = _get_split(ctx)
        y_ho = y_ho_s.to_numpy()

        if hasattr(ctx.baseline_run, "holdout_pred") and ctx.baseline_run.holdout_pred is not None:
            ho_pred = ctx.baseline_run.holdout_pred
            preds_ho = np.asarray(getattr(ho_pred, "y_pred", ho_pred), dtype=int)
        else:
            preds_ho = np.asarray(ctx.baseline_run.adapter.predict(X_ho), dtype=int)

        errors_ho = (preds_ho != y_ho).astype(int)
        n_ho_total = len(y_ho)

        testable_slices: list[Slice] = []

        for sl in candidates:
            mask_ho = get_slice_mask(X_ho, sl)
            n1 = int(np.sum(mask_ho))
            n2 = n_ho_total - n1

            sl.n_holdout = n1
            if n1 > 0:
                sl.error_holdout = float(np.mean(errors_ho[mask_ho]))
            else:
                sl.error_holdout = 0.0

            if n2 > 0:
                sl.complement_error_holdout = float(np.mean(errors_ho[~mask_ho]))
            else:
                sl.complement_error_holdout = 0.0

            sl.gap_holdout = sl.error_holdout - sl.complement_error_holdout

            if n1 < thresholds.slice_min_n or n2 < thresholds.slice_min_n:
                sl.testable = False
                sl.p_holdout = None
                continue

            sl.testable = True
            x1 = int(np.sum(errors_ho[mask_ho]))
            x2 = int(np.sum(errors_ho[~mask_ho]))

            # Certified Exact Unconditional Margin Test
            p_exact = exact_margin_test(
                x1=x1,
                n1=n1,
                x2=x2,
                n2=n2,
                delta=thresholds.slice_gap,
                n_grid=5000,
            )
            sl.p_holdout = p_exact
            testable_slices.append(sl)

        # If no slices met the minimum sample size on holdout
        if not testable_slices:
            return [
                HypothesisResult(
                    id="h4-subgroup",
                    type=HypothesisType.subgroup,
                    subtype=None,
                    status=Status.inconclusive,
                    evidence_strength=0.3,
                    trigger=trigger,
                    diagnostic_evidence_ids=[],
                    experiment_ids=[res.spec_id for res in results] if results else [],
                    decision=Decision(
                        rule_id="SG-D1",
                        rule_text=_RULE_SG_D1.text,
                        rules_profile=profile,
                        criteria_met={
                            "slice_confirmed": False,
                            "testable_sample_size": False,
                            "reweighting_effective": False,
                        },
                    ),
                    caveats=[
                        f"Candidate slices identified in training, but all had holdout sample size "
                        f"n < {thresholds.slice_min_n}. Statistical confirmation underpowered."
                    ],
                    recommended_actions=[
                        f"Collect more holdout samples for candidate slice '{candidates[0].name}' "
                        f"to reach n >= {thresholds.slice_min_n} before drawing conclusions."
                    ],
                )
            ]

        # Apply Benjamini-Yekutieli (BY) FDR control over the testable candidate family
        p_vals = [s.p_holdout if s.p_holdout is not None else 1.0 for s in testable_slices]
        q_vals = benjamini_yekutieli(p_vals)

        confirmed_slices: list[Slice] = []
        for sl, q in zip(testable_slices, q_vals, strict=True):
            sl.q_holdout = q
            if q <= thresholds.slice_fdr_q:
                sl.confirmed = True
                confirmed_slices.append(sl)
            else:
                sl.confirmed = False

        slice_confirmed = len(confirmed_slices) > 0

        exp = next((r for r in results if r.spec_id == "E-SG-2"), None)
        # Check reweighting experiment effectiveness
        reweighting_effective = False
        if results and slice_confirmed and exp is not None and exp.delta >= thresholds.delta_min:
            reweighting_effective = True

        criteria_met = {
            "slice_confirmed": slice_confirmed,
            "testable_sample_size": True,
            "reweighting_effective": reweighting_effective,
        }

        effect: EffectResult | None = None
        if exp is not None:
            metric_name = (
                ctx.config.primary_metric.value
                if ctx.config
                and hasattr(ctx.config, "primary_metric")
                and hasattr(ctx.config.primary_metric, "value")
                else "accuracy"
            )
            effect = EffectResult(
                metric=metric_name,
                baseline=float(exp.baseline_metric),
                intervention=float(exp.intervention_metric),
                delta=float(exp.delta),
                ci_low=float(exp.ci_low),
                ci_high=float(exp.ci_high),
                n=exp.n_eval,
            )
        elif confirmed_slices:
            top_conf = confirmed_slices[0]
            effect = EffectResult(
                metric="error_rate",
                baseline=float(top_conf.complement_error_holdout),
                intervention=float(top_conf.error_holdout),
                delta=float(top_conf.gap_holdout),
                ci_low=float(max(-1.0, top_conf.gap_holdout - 0.05)),
                ci_high=float(min(1.0, top_conf.gap_holdout + 0.05)),
                n=int(top_conf.n_holdout),
            )

        # Determine final status and subtype
        if slice_confirmed:
            status = Status.supported
            subtype = (
                "representation_related" if reweighting_effective else "intrinsic_or_unresolved"
            )
            top_conf = confirmed_slices[0]
            severity = (
                Severity.high
                if top_conf.gap_holdout >= 0.15
                else Severity.medium
                if top_conf.gap_holdout >= 0.08
                else Severity.low
            )
            strength = 0.9 if reweighting_effective else 0.75
            caveats = [
                f"Subgroup '{top_conf.name}' confirmed with holdout error gap "
                f"{top_conf.gap_holdout:.3f} under BY FDR control "
                f"(q = {top_conf.q_holdout:.4f} <= {thresholds.slice_fdr_q})."
            ]
            actions = [
                f"Inspect data collection for subgroup '{top_conf.name}'.",
                "Evaluate feature importance and subgroup-specific representation.",
            ]
        else:
            status = Status.rejected
            subtype = None
            severity = None
            strength = 0.2
            caveats = [
                "Candidate slices identified in training failed holdout confirmation under BY FDR "
                f"control (best q = {min(q_vals):.4f} > {thresholds.slice_fdr_q})."
            ]
            actions = ["No evidence of systematic subgroup underperformance."]

        return [
            HypothesisResult(
                id="h4-subgroup",
                type=HypothesisType.subgroup,
                subtype=subtype,
                status=status,
                severity=severity,
                evidence_strength=strength,
                trigger=trigger,
                diagnostic_evidence_ids=[],
                experiment_ids=[res.spec_id for res in results] if results else [],
                effect=effect,
                decision=Decision(
                    rule_id="SG-D1",
                    rule_text=_RULE_SG_D1.text,
                    rules_profile=profile,
                    criteria_met=criteria_met,
                ),
                caveats=caveats,
                recommended_actions=actions,
            )
        ]
