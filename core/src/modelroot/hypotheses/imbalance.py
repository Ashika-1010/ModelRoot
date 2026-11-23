"""H2 — Class imbalance hypothesis (M14).

Implements sub-claims:
- distortion: Evaluates whether metrics are distorted by imbalance (IM-D1).
- harm: Evaluates whether imbalance harms holdout performance, mitigated by interventions (IM-D2).
"""

from __future__ import annotations

import contextlib
import math
from typing import Any

import numpy as np
from sklearn.metrics import balanced_accuracy_score, precision_recall_fscore_support

from modelroot.assessment.metrics import metric_fn
from modelroot.evidence.rules import (
    RULE_IM_D1,
    RULE_IM_D2,
    decide_intervention_effect,
    evidence_strength,
    make_evidence_id,
)
from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.schemas.common import HypothesisType, RulesProfile, Status
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec
from modelroot.schemas.hypothesis import Decision, EffectResult, HypothesisResult, TriggerResult
from modelroot.stats.power import power_gate


def _safe_float(v: float) -> float:
    f = float(v)
    return f if math.isfinite(f) else 0.0


class ImbalanceHypothesis(Hypothesis):
    """H2 Class Imbalance and Metric Distortion hypothesis."""

    type: HypothesisType = HypothesisType.imbalance

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        """Imbalance applies to binary classification datasets."""
        y_tr = np.asarray(ctx.split.y_train)
        classes = np.unique(y_tr)
        if len(classes) != 2:
            return Applicability(
                applicable=False,
                reason=f"Imbalance requires binary classification (found {len(classes)} classes).",
            )
        return Applicability(applicable=True)

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        """Trigger IM-T1: minority share <= 0.20 and (acc-bacc gap >= 0.05 or min recall <= 0.5)."""
        y_tr = np.asarray(ctx.split.y_train, dtype=int)
        thresholds = ctx.thresholds

        n_total = len(y_tr)
        n_0 = int(np.sum(y_tr == 0))
        n_1 = int(np.sum(y_tr == 1))
        min_share = min(n_0, n_1) / max(1, n_total)
        min_class = 1 if n_1 < n_0 else 0
        maj_share = 1.0 - min_share

        # Evaluate baseline holdout predictions
        ho_true = np.asarray(ctx.baseline_run.holdout_pred.y_true, dtype=int)
        ho_pred = np.asarray(ctx.baseline_run.holdout_pred.y_pred, dtype=int)
        ho_proba = np.asarray(ctx.baseline_run.holdout_pred.y_proba, dtype=float)

        acc = float(np.mean(ho_true == ho_pred))
        bacc = float(balanced_accuracy_score(ho_true, ho_pred))
        gap = acc - bacc

        # Minority class recall
        _p, r, _f, _s = precision_recall_fscore_support(
            ho_true, ho_pred, labels=[min_class], zero_division=0
        )
        min_recall = float(r[0]) if len(r) > 0 else 0.0

        # Optional PR-AUC and ROC-AUC
        auc_fn = metric_fn("roc_auc")
        roc_auc = float(auc_fn(ho_true, ho_pred, ho_proba))

        # Check trigger condition
        min_share_thresh = thresholds.minority_share if thresholds else 0.20
        acc_bacc_thresh = thresholds.acc_bacc_gap if thresholds else 0.05
        fired = bool(
            (min_share <= min_share_thresh) and ((gap >= acc_bacc_thresh) or (min_recall <= 0.5))
        )

        vals: dict[str, float | str | bool | int | None] = {
            "minority_share": _safe_float(min_share),
            "majority_share": _safe_float(maj_share),
            "acc_bacc_gap": _safe_float(gap),
            "minority_recall": _safe_float(min_recall),
            "roc_auc": _safe_float(roc_auc),
        }

        return TriggerResult(fired=fired, rule_id="IM-T1", values=vals)

    def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list[ExperimentSpec]:
        """Plan E-IM-1/2/3 selection via inner-CV (imbalance_best)."""
        if not trigger.fired:
            return []

        thresholds = ctx.thresholds
        n_boot = thresholds.n_boot if thresholds else 1000
        seed = ctx.seeds.spawn("imbalance:harm") if ctx.seeds else ctx.config.seed

        return [
            ExperimentSpec(
                id="exp-imbalance-harm",
                hypothesis_id="h2-imbalance",
                name="imbalance_best",
                intervention="imbalance_best",
                params={},
                metric=ctx.config.primary_metric.value,
                eval_on=["holdout"],
                n_boot=n_boot,
                repeats=0,
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
        """Produce verdicts for sub-claims distortion and harm."""
        thresholds = ctx.thresholds
        profile = ctx.config.rules_profile or RulesProfile.full

        # Extract trigger values
        min_share = float(trigger.values.get("minority_share", 0.5) or 0.5)
        maj_share = float(trigger.values.get("majority_share", 0.5) or 0.5)
        acc_bacc_gap = float(trigger.values.get("acc_bacc_gap", 0.0) or 0.0)
        min_recall = float(trigger.values.get("minority_recall", 1.0) or 1.0)
        roc_auc = float(trigger.values.get("roc_auc", 0.5) or 0.5)

        ho_true = np.asarray(ctx.baseline_run.holdout_pred.y_true, dtype=int)
        ho_pred = np.asarray(ctx.baseline_run.holdout_pred.y_pred, dtype=int)
        acc = float(np.mean(ho_true == ho_pred))

        diag_ev_ids: list[str] = [
            make_evidence_id("minority_share", min_share, "baseline_assessment", None),
            make_evidence_id("majority_share", maj_share, "baseline_assessment", None),
            make_evidence_id("acc_bacc_gap", acc_bacc_gap, "baseline_assessment", None),
            make_evidence_id("minority_recall", min_recall, "baseline_assessment", None),
            make_evidence_id("roc_auc", roc_auc, "baseline_assessment", None),
        ]

        maj_margin = thresholds.majority_margin if thresholds else 0.02
        acc_bacc_thresh = thresholds.acc_bacc_gap if thresholds else 0.05
        min_share_thresh = thresholds.minority_share if thresholds else 0.20

        # --- Sub-claim 1: distortion (IM-D1) ---
        acc_near_maj = acc <= (maj_share + maj_margin)
        gap_high = acc_bacc_gap >= acc_bacc_thresh

        dist_criteria = {
            "acc_near_majority": acc_near_maj,
            "acc_bacc_gap": gap_high,
        }

        if not trigger.fired:
            dist_status = Status.not_triggered
            dist_strength = 0.0
        elif acc_near_maj or gap_high:
            dist_status = Status.supported
            dist_strength = evidence_strength(RULE_IM_D1, dist_criteria)
        elif min_share <= min_share_thresh:
            dist_status = Status.weakly_supported
            dist_strength = 0.5
        else:
            dist_status = Status.rejected
            dist_strength = 0.0

        dist_result = HypothesisResult(
            id="h2-imbalance-distortion",
            type=HypothesisType.imbalance,
            subtype="distortion",
            status=dist_status,
            evidence_strength=_safe_float(dist_strength),
            trigger=trigger,
            diagnostic_evidence_ids=diag_ev_ids,
            experiment_ids=[],
            effect=None,
            decision=Decision(
                rule_id=RULE_IM_D1.id,
                rule_text=RULE_IM_D1.text,
                rules_profile=profile,
                criteria_met=dist_criteria,
            ),
            caveats=[
                "Accuracy may reflect majority class prevalence rather than model discernment."
            ],
            recommended_actions=[
                "Use balanced_accuracy or macro_f1 as primary evaluation metric.",
                "Review class-specific confusion matrix and minority precision/recall.",
            ]
            if dist_status in (Status.supported, Status.weakly_supported)
            else [],
        )

        # --- Sub-claim 2: harm (IM-D2) ---
        if not trigger.fired:
            harm_result = HypothesisResult(
                id="h2-imbalance-harm",
                type=HypothesisType.imbalance,
                subtype="harm",
                status=Status.not_triggered,
                evidence_strength=0.0,
                trigger=trigger,
                diagnostic_evidence_ids=diag_ev_ids,
                experiment_ids=[],
                effect=None,
                decision=Decision(
                    rule_id=RULE_IM_D2.id,
                    rule_text=RULE_IM_D2.text,
                    rules_profile=profile,
                    criteria_met={},
                ),
                caveats=[],
                recommended_actions=[],
            )
            return [dist_result, harm_result]

        if not results:
            harm_result = HypothesisResult(
                id="h2-imbalance-harm",
                type=HypothesisType.imbalance,
                subtype="harm",
                status=Status.not_triggered if not trigger.fired else Status.inconclusive,
                evidence_strength=0.0,
                trigger=trigger,
                diagnostic_evidence_ids=diag_ev_ids,
                experiment_ids=[],
                effect=None,
                decision=Decision(
                    rule_id=RULE_IM_D2.id,
                    rule_text=RULE_IM_D2.text,
                    rules_profile=profile,
                    criteria_met={},
                ),
                caveats=[]
                if not trigger.fired
                else ["No experiment results available (no-experiments mode or budget limit)."],
                recommended_actions=[],
            )
            return [dist_result, harm_result]

        exp: ExperimentResult = results[0]
        exp_ids = [exp.spec_id]

        # Check minority sample size limitation
        y_tr = np.asarray(ctx.split.y_train, dtype=int)
        n_min_tr = int(min(np.sum(y_tr == 0), np.sum(y_tr == 1)))
        min_oof_thresh = thresholds.minority_oof_min if thresholds else 30

        # Parse notes from intervention outcome
        notes = getattr(exp, "notes", [])
        selected_intervention = "unknown"
        threshold_prior_issue = False
        delta_auc = 0.0

        for n_str in notes:
            if n_str.startswith("selected_intervention="):
                selected_intervention = n_str[len("selected_intervention=") :]
            elif n_str.startswith("threshold_prior_issue="):
                threshold_prior_issue = n_str[len("threshold_prior_issue=") :].lower() == "true"
            elif n_str.startswith("delta_auc="):
                with contextlib.suppress(ValueError):
                    delta_auc = float(n_str[len("delta_auc=") :])

        # Power gate
        n_eval = exp.n_eval
        n_pos = int(np.sum(ho_true == 1))
        ci_width = exp.ci_high - exp.ci_low
        gate = power_gate(
            n_holdout=n_eval,
            n_pos=n_pos,
            ci_width=ci_width,
            holdout_min_n=thresholds.holdout_min_n if thresholds else 200,
            holdout_min_pos=thresholds.holdout_min_pos if thresholds else 30,
            ci_width_max=thresholds.ci_width_max if thresholds else 0.10,
        )

        adequate_min_oof = n_min_tr >= min_oof_thresh

        if not adequate_min_oof:
            harm_status = Status.inconclusive
            harm_criteria = {
                "delta_min": exp.delta >= (thresholds.delta_min if thresholds else 0.02),
                "ci_excludes_zero": exp.ci_low > 0,
                "adequately_powered": gate.ok,
                "minority_oof_adequate": False,
            }
            harm_strength = 0.0
            caveats = [
                f"Minority training samples ({n_min_tr}) < {min_oof_thresh}; "
                "threshold tuning and sampling variance limit confidence."
            ]
        else:
            harm_status, harm_criteria = decide_intervention_effect(
                delta=exp.delta,
                ci_low=exp.ci_low,
                ci_high=exp.ci_high,
                gate_ok=gate.ok,
                extra_criteria={"minority_oof_adequate": True},
                thresholds=thresholds,
                profile=profile,
            )
            harm_strength = evidence_strength(RULE_IM_D2, harm_criteria)
            caveats = []

        # Nuance flag: ROC-AUC unchanged while primary metric improves
        if threshold_prior_issue and harm_status in (Status.supported, Status.weakly_supported):
            caveats.append(
                f"ROC-AUC unchanged (ΔAUC={delta_auc:+.3f}) while "
                f"{ctx.config.primary_metric.value} improved. The base model's ranking is "
                "preserved; the issue is primarily an uncalibrated classification threshold."
            )

        effect = EffectResult(
            metric=ctx.config.primary_metric.value,
            baseline=_safe_float(exp.baseline_metric),
            intervention=_safe_float(exp.intervention_metric),
            delta=_safe_float(exp.delta),
            ci_low=_safe_float(exp.ci_low),
            ci_high=_safe_float(exp.ci_high),
            n=n_eval,
        )

        recommended = []
        if harm_status in (Status.supported, Status.weakly_supported):
            if selected_intervention == "threshold_tune" or threshold_prior_issue:
                recommended.append(
                    "Adjust the classification decision threshold on predicted "
                    "probabilities without retraining."
                )
            elif selected_intervention == "class_weight":
                recommended.append(
                    "Apply balanced class weighting (class_weight='balanced') during training."
                )
            elif selected_intervention == "oversample_train":
                recommended.append("Oversample the minority class in the training split.")

        harm_result = HypothesisResult(
            id="h2-imbalance-harm",
            type=HypothesisType.imbalance,
            subtype="harm",
            status=harm_status,
            evidence_strength=_safe_float(harm_strength),
            trigger=trigger,
            diagnostic_evidence_ids=diag_ev_ids,
            experiment_ids=exp_ids,
            effect=effect,
            decision=Decision(
                rule_id=RULE_IM_D2.id,
                rule_text=RULE_IM_D2.text,
                rules_profile=profile,
                criteria_met=harm_criteria,
            ),
            caveats=caveats,
            recommended_actions=recommended,
        )

        return [dist_result, harm_result]
