"""Distribution Shift Hypothesis (H5: M20).

Tests whether dataset drift between reference (train/holdout) and current production
data causes severe model performance degradation, substantiated by multidimensional
domain classification, univariate drift metrics (KS / Chi-square / PSI), and
dose-response error analysis across quantile bins of the top drifted feature.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.stats as stats

from modelroot.config.thresholds import Thresholds
from modelroot.evidence.rules import Criterion, Rule
from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.schemas.common import HypothesisType, RulesProfile, Severity, Status
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec
from modelroot.schemas.hypothesis import Decision, EffectResult, HypothesisResult, TriggerResult
from modelroot.stats.drift import DriftReport, compute_drift_report

_RULE_SH_D1 = Rule(
    id="SH-D1",
    text=(
        "Supported if dataset drift is statistically confirmed, performance degrades "
        "by >= delta_min on current data, and dose-response or intervention recovery "
        "is confirmed. Weakly supported if drift and degradation occur without dose-response. "
        "Rejected if current performance does not drop."
    ),
    criteria=[
        Criterion(
            id="drift_confirmed",
            weight=2.0,
            description="Domain classifier AUC >= drift_auc_min (p < drift_p_max) or feature drift",
        ),
        Criterion(
            id="performance_degraded",
            weight=2.0,
            description="Current performance drops by >= delta_min relative to holdout",
        ),
        Criterion(
            id="dose_response_or_recovery",
            weight=1.0,
            description="Dose-response correlation or intervention recovery >= delta_min/2",
        ),
    ],
)


class ShiftHypothesis(Hypothesis):
    """H5: Distribution Shift Hypothesis."""

    type: HypothesisType = HypothesisType.shift

    def __init__(self) -> None:
        self._cached_report: DriftReport | None = None
        self._cached_ctx_id: int | None = None

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        """Applicable only when a current production dataset is provided."""
        has_curr = False
        if hasattr(ctx.split, "current") and ctx.split.current is not None:
            if hasattr(ctx.split.current, "X") and len(ctx.split.current.X) > 0:
                has_curr = True
        elif (
            hasattr(ctx.split, "X_current")
            and ctx.split.X_current is not None
            and len(ctx.split.X_current) > 0
        ):
            has_curr = True

        if not has_curr:
            return Applicability(
                applicable=False,
                reason="No current production dataset provided for distribution shift analysis.",
            )

        return Applicability(applicable=True)

    def _get_dfs(self, ctx: HypothesisContext) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Extract reference (training) and current DataFrames."""
        if hasattr(ctx.split, "train"):
            df_ref = ctx.split.train.X
            df_cur = ctx.split.current.X
        else:
            df_ref = ctx.split.X_train
            df_cur = ctx.split.X_current
        return df_ref, df_cur

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        """Evaluate dataset drift using domain classification and univariate statistics."""
        # Always reset cache on trigger invocation
        self._cached_report = None
        self._cached_ctx_id = None

        if not self.applicable(ctx).applicable:
            return TriggerResult(
                fired=False,
                rule_id="SH-T1",
                values={"fired": False, "reason": "not_applicable"},
            )

        thresholds = ctx.thresholds or Thresholds()
        df_ref, df_cur = self._get_dfs(ctx)

        target_col = ctx.config.target if ctx.config else None
        seed = ctx.seed_for("shift", "domain_classifier") if hasattr(ctx, "seed_for") else 0

        # Fast 20 permutations for trigger check
        report = compute_drift_report(
            df_ref=df_ref,
            df_cur=df_cur,
            target_column=target_col,
            seed=seed,
            n_permutations=20,
        )

        self._cached_report = report
        self._cached_ctx_id = id(ctx)

        # Domain classifier condition
        domain_fired = (
            report.domain_classifier.auc >= thresholds.drift_auc_min
            and report.domain_classifier.p_value < thresholds.drift_p_max
        )

        # Univariate feature drift condition
        feat_fired = False
        top_feat = report.top_drifted_features[0] if report.top_drifted_features else ""
        top_psi = 0.0
        top_stat = 0.0

        for _feat, u in report.univariate.items():
            if u.psi >= thresholds.drift_psi_min:
                feat_fired = True
            if (
                u.feature_type == "numeric"
                and u.statistic_value >= thresholds.drift_ks_min
                and u.p_value < 0.05
            ):
                feat_fired = True
            if u.feature_type == "categorical" and u.p_value < 0.05 and u.psi >= 0.10:
                feat_fired = True

        if top_feat and top_feat in report.univariate:
            top_psi = report.univariate[top_feat].psi
            top_stat = report.univariate[top_feat].statistic_value

        fired = domain_fired or feat_fired

        values: dict[str, float | str | bool | int | None] = {
            "fired": fired,
            "domain_auc": float(report.domain_classifier.auc),
            "domain_p_value": float(report.domain_classifier.p_value),
            "top_feature": str(top_feat),
            "top_psi": float(top_psi),
            "top_stat": float(top_stat),
            "n_reference": len(df_ref),
            "n_current": len(df_cur),
        }

        return TriggerResult(fired=fired, rule_id="SH-T1", values=values)

    def plan(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
    ) -> list[ExperimentSpec]:
        """Plan mechanism experiments (feature ablation, quantile alignment) on current data."""
        if not trigger.fired:
            return []

        report = self._cached_report
        if report is None or self._cached_ctx_id != id(ctx):
            df_ref, df_cur = self._get_dfs(ctx)
            target_col = ctx.config.target if ctx.config else None
            seed = ctx.seed_for("shift", "domain_classifier") if hasattr(ctx, "seed_for") else 0
            report = compute_drift_report(
                df_ref=df_ref,
                df_cur=df_cur,
                target_column=target_col,
                seed=seed,
                n_permutations=20,
            )
            self._cached_report = report
            self._cached_ctx_id = id(ctx)

        specs: list[ExperimentSpec] = []
        if ctx.config and getattr(ctx.config, "run_experiments", False):
            top_feat = report.top_drifted_features[0] if report.top_drifted_features else ""
            metric_str = (
                ctx.config.primary_metric.value
                if hasattr(ctx.config, "primary_metric")
                and hasattr(ctx.config.primary_metric, "value")
                else "accuracy"
            )
            seed_val = ctx.seed_for("experiment", "shift_ablate") if hasattr(ctx, "seed_for") else 0

            # Experiment 1: Ablate top drifted feature
            if top_feat:
                specs.append(
                    ExperimentSpec(
                        id="E-SH-1",
                        hypothesis_id="h5-shift",
                        name=f"Ablate top drifted feature '{top_feat}'",
                        intervention="ablate_drifted",
                        params={"feature": top_feat},
                        metric=metric_str,
                        eval_on=["current"],
                        seed=seed_val,
                        budget_s=30.0,
                    )
                )

            # Experiment 2: Quantile align top numeric features
            top_nums = [
                f
                for f in report.top_drifted_features
                if report.univariate[f].feature_type == "numeric"
            ][:3]
            if top_nums:
                specs.append(
                    ExperimentSpec(
                        id="E-SH-2",
                        hypothesis_id="h5-shift",
                        name="Quantile align top drifted numeric features",
                        intervention="quantile_align",
                        params={"features": ",".join(top_nums)},
                        metric=metric_str,
                        eval_on=["current"],
                        seed=seed_val + 1,
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
        """Confirm distribution shift harm via current degradation and dose-response."""
        thresholds = ctx.thresholds or Thresholds()
        profile = (
            ctx.config.rules_profile
            if ctx.config and hasattr(ctx.config, "rules_profile")
            else RulesProfile.full
        )

        if not trigger.fired:
            return [
                HypothesisResult(
                    id="h5-shift",
                    type=HypothesisType.shift,
                    subtype=None,
                    status=Status.not_triggered,
                    evidence_strength=0.0,
                    trigger=trigger,
                    diagnostic_evidence_ids=[],
                    experiment_ids=[],
                    decision=Decision(
                        rule_id="SH-D1",
                        rule_text=_RULE_SH_D1.text,
                        rules_profile=profile,
                        criteria_met={
                            "drift_confirmed": False,
                            "performance_degraded": False,
                            "dose_response_or_recovery": False,
                        },
                    ),
                    caveats=["No statistically significant dataset drift detected."],
                    recommended_actions=[
                        "Continue standard distribution monitoring in production."
                    ],
                )
            ]

        # 1. Measure baseline performance degradation on current data
        df_ref, df_cur = self._get_dfs(ctx)
        if hasattr(ctx.split, "holdout"):
            y_ho = ctx.split.holdout.y.to_numpy()
            y_cur = (
                ctx.split.current.y.to_numpy() if ctx.split.current is not None else np.array([])
            )
            X_ho = ctx.split.holdout.X
        else:
            y_ho = ctx.split.y_holdout.to_numpy()
            y_cur = (
                ctx.split.y_current.to_numpy() if ctx.split.y_current is not None else np.array([])
            )
            X_ho = ctx.split.X_holdout

        if hasattr(ctx.baseline_run, "holdout_pred") and ctx.baseline_run.holdout_pred is not None:
            ho_pred = ctx.baseline_run.holdout_pred
            preds_ho = np.asarray(getattr(ho_pred, "y_pred", ho_pred), dtype=int)
        else:
            preds_ho = np.asarray(ctx.baseline_run.adapter.predict(X_ho), dtype=int)

        if hasattr(ctx.baseline_run, "current_pred") and ctx.baseline_run.current_pred is not None:
            cur_pred = ctx.baseline_run.current_pred
            preds_cur = np.asarray(getattr(cur_pred, "y_pred", cur_pred), dtype=int)
        else:
            preds_cur = np.asarray(ctx.baseline_run.adapter.predict(df_cur), dtype=int)

        score_ho = float(np.mean(preds_ho == y_ho))
        score_cur = float(np.mean(preds_cur == y_cur))
        perf_drop = score_ho - score_cur  # Positive drop means current performs worse

        # 2. Dose-response analysis on top drifted feature
        report = self._cached_report
        top_feat = report.top_drifted_features[0] if report and report.top_drifted_features else ""

        dose_significant = False
        dose_rho = 0.0
        dose_p = 1.0

        if top_feat and top_feat in df_cur.columns:
            feat_cur = df_cur[top_feat]
            if pd.api.types.is_numeric_dtype(feat_cur) and len(df_cur) >= thresholds.dose_min_rows:
                # Bin into 5 or 10 quantile bins
                n_bins = 10 if len(df_cur) >= 1000 else 5
                err_cur = (preds_cur != y_cur).astype(int)

                try:
                    bins, bin_edges = pd.qcut(feat_cur, q=n_bins, retbins=True, duplicates="drop")
                    unique_bins = pd.unique(bins)
                    if len(unique_bins) >= thresholds.dose_min_bins:
                        # Error rate in each quantile bin
                        per_bin_errs: list[float] = []
                        for b_cat in sorted(unique_bins, key=lambda b: b.left):
                            mask = (bins == b_cat).to_numpy(dtype=bool)
                            if np.sum(mask) > 0:
                                per_bin_errs.append(float(np.mean(err_cur[mask])))

                        if len(per_bin_errs) >= thresholds.dose_min_bins:
                            # Monotonic relationship between feature severity and error rate
                            err_arr = np.array(per_bin_errs)
                            if np.all(err_arr == err_arr[0]):
                                dose_rho = 0.0
                                dose_p = 1.0
                            else:
                                rho, p_val = stats.spearmanr(np.arange(len(per_bin_errs)), err_arr)
                                dose_rho = float(rho) if not np.isnan(rho) else 0.0
                                dose_p = float(p_val) if not np.isnan(p_val) else 1.0

                            if (
                                dose_rho > thresholds.dose_rho_min
                                and dose_p < thresholds.dose_p_max
                            ):
                                dose_significant = True
                except Exception:
                    dose_significant = False

        # 3. Check intervention recovery
        recovery_effective = False
        exp = next((r for r in results if r.spec_id in ("E-SH-1", "E-SH-2")), None)
        if exp is not None and exp.delta >= (thresholds.delta_min / 2.0):
            recovery_effective = True

        # 4. Synthesize decision
        drift_confirmed = trigger.fired
        performance_degraded = perf_drop >= thresholds.delta_min
        dose_or_recovery = dose_significant or recovery_effective

        criteria_met = {
            "drift_confirmed": drift_confirmed,
            "performance_degraded": performance_degraded,
            "dose_response_or_recovery": dose_or_recovery,
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
        else:
            effect = EffectResult(
                metric="accuracy",
                baseline=float(score_ho),
                intervention=float(score_cur),
                delta=float(-perf_drop),
                ci_low=float(-perf_drop - 0.05),
                ci_high=float(-perf_drop + 0.05),
                n=len(y_cur),
            )

        if drift_confirmed and performance_degraded and dose_or_recovery:
            status = Status.supported
            strength = 0.9 if (dose_significant and recovery_effective) else 0.8
            severity = (
                Severity.high
                if perf_drop >= 0.10
                else Severity.medium
                if perf_drop >= 0.05
                else Severity.low
            )
            caveats = [
                "Production distribution shift confirmed "
                f"(domain AUC={trigger.values.get('domain_auc')}) "
                f"inducing accuracy degradation of {perf_drop:.3f} >= {thresholds.delta_min}."
            ]
            actions = [
                f"Retrain baseline model incorporating recent data or ablate feature '{top_feat}'.",
                "Evaluate feature store pipelines for upstream schema or population changes.",
            ]
        elif drift_confirmed and performance_degraded:
            status = Status.weakly_supported
            strength = 0.6
            severity = Severity.low
            caveats = [
                f"Performance dropped by {perf_drop:.3f} on current data alongside shift, "
                "but dose-response correlation was not statistically significant."
            ]
            actions = ["Monitor current data stream and evaluate domain adaptation strategies."]
        else:
            status = Status.rejected
            strength = 0.2
            severity = None
            caveats = [
                "Dataset shift detected, but performance on current data did not degrade "
                f"significantly (drop = {perf_drop:.3f} < {thresholds.delta_min}). "
                "Benign covariate shift."
            ]
            actions = ["No intervention required; current model predictions remain robust."]

        return [
            HypothesisResult(
                id="h5-shift",
                type=HypothesisType.shift,
                subtype=None,
                status=status,
                severity=severity,
                evidence_strength=strength,
                trigger=trigger,
                diagnostic_evidence_ids=[],
                experiment_ids=[res.spec_id for res in results] if results else [],
                effect=effect,
                decision=Decision(
                    rule_id="SH-D1",
                    rule_text=_RULE_SH_D1.text,
                    rules_profile=profile,
                    criteria_met=criteria_met,
                ),
                caveats=caveats,
                recommended_actions=actions,
            )
        ]
