"""H3 — Leakage hypotheses (M15 Duplicate & Identifier, M16 Target-Proxy).

Implements:
- H3a: Duplicate/overlap leakage (leakage_duplicate, subtype duplicate, LK-A-T1, LK-A-D1)
- H3b: Identifier/row-order leakage (leakage_identifier, subtype identifier, LK-B-T1, LK-B-D1)
- H3c: Target-proxy leakage (leakage_target_proxy, subtype target_proxy, LK-C-T1, LK-C-D1)
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from modelroot.assessment.metrics import metric_fn, needs_proba
from modelroot.config.thresholds import Thresholds
from modelroot.evidence.rules import (
    RULE_LK_A_D1,
    RULE_LK_B_D1,
    decide_intervention_effect,
    evidence_strength,
    make_evidence_id,
    severity,
)
from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.models.registry import get_model
from modelroot.schemas.common import HypothesisType, SplitMode, Status
from modelroot.schemas.hypothesis import Decision, EffectResult, HypothesisResult, TriggerResult
from modelroot.stats.bootstrap import BootstrapResult, paired_bootstrap
from modelroot.stats.power import power_gate
from modelroot.stats.subset_bootstrap import paired_subset_bootstrap

# Constants for H3b Identifier Leakage
ID_LABEL_RHO: float = 0.2
ID_LABEL_P: float = 0.01
IMPORTANCE_REPEATS: int = 5
MAX_EVIDENCE_FEATURES: int = 5


def _safe_float(v: float) -> float:
    f = float(v)
    return f if math.isfinite(f) else 0.0


def row_hashes(X: pd.DataFrame) -> np.ndarray:
    """Return uint64 hashes of complete feature rows.

    Normalizes numerics to float64 and categoricals to strings.
    Preserves consistent NaN hashing. Does not mutate input.
    Detects exact duplicates only.
    """
    df = X.copy()
    for col in df.columns:
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            df[col] = s.astype(np.float64)
        elif isinstance(s.dtype, pd.CategoricalDtype):
            df[col] = s.astype(object)
        elif not pd.api.types.is_string_dtype(s) and not pd.api.types.is_object_dtype(s):
            df[col] = s.astype(str)
    return np.asarray(
        pd.util.hash_pandas_object(df, index=False).to_numpy(dtype=np.uint64),
        dtype=np.uint64,
    )


def permutation_importance_share(
    ctx: HypothesisContext,
    *,
    n_repeats: int = IMPORTANCE_REPEATS,
) -> dict[str, float]:
    """Compute permutation importance shares on holdout data with fitted baseline model.

    Permutes one feature column at a time on copies.
    Importance = baseline score - mean permuted score.
    Normalizes positive importance values into shares; all-zero if no positive importance.
    Deterministic seed via ctx.seed_for("leakage", "importance").
    No refitting or input mutation.
    """
    X_ho = ctx.split.X_holdout
    y_ho = np.asarray(ctx.split.y_holdout, dtype=int)
    metric = ctx.config.primary_metric.value
    fn = metric_fn(metric)
    use_proba = needs_proba(metric)

    p_base = np.asarray(ctx.baseline_run.holdout_pred.y_pred, dtype=int)
    pr_base: np.ndarray = (
        np.asarray(ctx.baseline_run.holdout_pred.y_proba, dtype=float)
        if use_proba
        else np.zeros(len(y_ho), dtype=float)
    )
    base_score = float(fn(y_ho, p_base, pr_base))

    seed = ctx.seed_for("leakage", "importance")
    rng = np.random.default_rng(seed)

    importances: dict[str, float] = {}
    for col in X_ho.columns:
        scores_rep: list[float] = []
        for _ in range(n_repeats):
            X_perm = X_ho.copy()
            perm_idx = rng.permutation(len(X_perm))
            X_perm[col] = X_ho[col].iloc[perm_idx].to_numpy()
            pred_perm = np.asarray(ctx.baseline_run.adapter.predict(X_perm), dtype=int)
            proba_perm: np.ndarray = (
                np.asarray(ctx.baseline_run.adapter.predict_proba(X_perm)[:, 1], dtype=float)
                if use_proba
                else np.zeros(len(y_ho), dtype=float)
            )
            scores_rep.append(float(fn(y_ho, pred_perm, proba_perm)))
        importances[col] = base_score - float(np.mean(scores_rep))

    pos_imp = {col: max(0.0, imp) for col, imp in importances.items()}
    tot_pos = sum(pos_imp.values())
    if tot_pos > 0:
        return {col: float(pos_imp[col] / tot_pos) for col in X_ho.columns}
    return dict.fromkeys(X_ho.columns, 0.0)


# ---------------------------------------------------------------------------
# H3a: Duplicate / Overlap Leakage
# ---------------------------------------------------------------------------


class LeakageDuplicateHypothesis(Hypothesis):
    """H3a Duplicate/overlap leakage hypothesis."""

    type: HypothesisType = HypothesisType.leakage
    name: str = "leakage_duplicate"

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        """Applies when both train and holdout features exist."""
        if ctx.split.X_train is None or ctx.split.X_holdout is None:
            return Applicability(applicable=False, reason="Missing train or holdout split.")
        return Applicability(applicable=True)

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        """Trigger LK-A-T1: duplicate holdout fraction >= thresholds.dup_frac."""
        tr_hashes = set(row_hashes(ctx.split.X_train))
        ho_hashes = row_hashes(ctx.split.X_holdout)

        is_dup = np.array([h in tr_hashes for h in ho_hashes], dtype=bool)
        dup_count = int(np.sum(is_dup))
        n_ho = len(ho_hashes)
        dup_fraction = dup_count / max(1, n_ho)

        thresh = ctx.thresholds.dup_frac if ctx.thresholds else 0.01
        fired = bool(dup_fraction >= thresh)

        vals: dict[str, float | str | bool | int | None] = {
            "dup_fraction": _safe_float(dup_fraction),
            "dup_count": dup_count,
            "n_holdout": n_ho,
        }
        return TriggerResult(fired=fired, rule_id="LK-A-T1", values=vals)

    def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list[Any]:
        """Returns empty list per specification."""
        return []

    def evaluate(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
        results: list[Any],
    ) -> list[HypothesisResult]:
        """Evaluate duplicate leakage effect via paired subset bootstrap."""
        rule = RULE_LK_A_D1
        profile = ctx.config.rules_profile
        hyp_id = self.name
        caveats = ["Exact duplicates only; real-data duplicates are not necessarily errors."]
        actions = ["Remove duplicate rows before splitting; re-evaluate."]

        if not trigger.fired:
            decision = Decision(
                rule_id=rule.id,
                rule_text=rule.text,
                rules_profile=profile,
                criteria_met={"dup_frac_gte_thresh": False},
            )
            return [
                HypothesisResult(
                    id=hyp_id,
                    type=self.type,
                    subtype="duplicate",
                    status=Status.not_triggered,
                    evidence_strength=0.0,
                    severity=None,
                    trigger=trigger,
                    decision=decision,
                    caveats=caveats,
                    recommended_actions=actions,
                )
            ]

        # Recompute mask on holdout
        tr_hashes = set(row_hashes(ctx.split.X_train))
        ho_hashes = row_hashes(ctx.split.X_holdout)
        is_dup = np.array([h in tr_hashes for h in ho_hashes], dtype=bool)
        non_dup_mask = ~is_dup

        ho_true = np.asarray(ctx.split.y_holdout, dtype=int)
        ho_pred = np.asarray(ctx.baseline_run.holdout_pred.y_pred, dtype=int)
        raw_proba = ctx.baseline_run.holdout_pred.y_proba
        ho_proba = (
            np.asarray(raw_proba, dtype=float)
            if raw_proba is not None and len(raw_proba) == len(ho_true)
            else None
        )

        n_non_dup = int(np.sum(non_dup_mask))
        classes_non_dup: np.ndarray = (
            np.unique(ho_true[non_dup_mask]) if n_non_dup > 0 else np.array([], dtype=int)
        )

        if n_non_dup < 2 or len(classes_non_dup) < 2:
            decision = Decision(
                rule_id=rule.id,
                rule_text=rule.text,
                rules_profile=profile,
                criteria_met={"dup_frac_gte_thresh": True, "adequately_powered": False},
            )
            return [
                HypothesisResult(
                    id=hyp_id,
                    type=self.type,
                    subtype="duplicate",
                    status=Status.inconclusive,
                    evidence_strength=0.0,
                    severity=None,
                    trigger=trigger,
                    decision=decision,
                    caveats=[
                        *caveats,
                        "Deduplicated holdout subset has fewer than 2 samples "
                        "or lacks class representation.",
                    ],
                    recommended_actions=actions,
                )
            ]

        metric = ctx.config.primary_metric.value

        boot = paired_subset_bootstrap(
            y_true=ho_true,
            pred=ho_pred,
            mask=non_dup_mask,
            metric=metric,
            n_boot=ctx.thresholds.n_boot if ctx.thresholds else 1000,
            seed=ctx.seed_for("leakage", "duplicate"),
            alpha=ctx.thresholds.alpha if ctx.thresholds else 0.05,
            y_proba=ho_proba,
        )

        n_sub = int(np.sum(non_dup_mask))
        n_pos_sub = int(np.sum(ho_true[non_dup_mask] == 1))
        ci_w = float(boot.ci_high - boot.ci_low)
        t = ctx.thresholds
        gate = power_gate(
            n_holdout=n_sub,
            n_pos=n_pos_sub,
            ci_width=ci_w,
            holdout_min_n=t.holdout_min_n if t else 200,
            holdout_min_pos=t.holdout_min_pos if t else 30,
            ci_width_max=t.ci_width_max if t else 0.10,
        )

        status, criteria_met = decide_intervention_effect(
            delta=boot.delta,
            ci_low=boot.ci_low,
            ci_high=boot.ci_high,
            gate_ok=gate.ok,
            extra_criteria={"dup_frac_gte_thresh": True},
            thresholds=ctx.thresholds,
            profile=profile,
        )

        strength = evidence_strength(rule, criteria_met)
        sev = (
            severity(boot.delta, ctx.thresholds)
            if status in (Status.supported, Status.weakly_supported)
            else None
        )

        effect = EffectResult(
            metric=metric,
            baseline=float(boot.metric_a),
            intervention=float(boot.metric_b),
            delta=float(boot.delta),
            ci_low=float(boot.ci_low),
            ci_high=float(boot.ci_high),
            n=len(ho_true),
        )

        dup_frac = float(trigger.values.get("dup_fraction", 0.0) or 0.0)
        ev_ids = [
            make_evidence_id("dup_fraction", dup_frac, "hash_overlap", None),
            make_evidence_id("full_score", float(boot.metric_b), "metric", None),
            make_evidence_id("deduplicated_score", float(boot.metric_a), "metric", None),
            make_evidence_id("inflation", float(boot.delta), "paired_subset_bootstrap", None),
        ]

        decision = Decision(
            rule_id=rule.id,
            rule_text=rule.text,
            rules_profile=profile,
            criteria_met=criteria_met,
        )

        return [
            HypothesisResult(
                id=hyp_id,
                type=self.type,
                subtype="duplicate",
                status=status,
                evidence_strength=strength,
                severity=sev,
                trigger=trigger,
                diagnostic_evidence_ids=ev_ids,
                effect=effect,
                decision=decision,
                caveats=caveats,
                recommended_actions=actions,
            )
        ]


# ---------------------------------------------------------------------------
# H3b: Identifier / Row-order Leakage
# ---------------------------------------------------------------------------

_NA_STRINGS: set[str] = {
    "",
    "nan",
    "NaN",
    "null",
    "NA",
    "None",
    "<NA>",
    "na",
    "N/A",
    "n/a",
}


def _is_continuous_numeric(s: pd.Series) -> bool:
    """Safely determine if a series represents continuous numeric values.

    Returns True for native or string-encoded continuous numeric features with
    fractional components. Returns False for integers, integer-like sequences
    (such as row IDs), and non-numeric string identifiers.

    For object-dtype columns the check is performed on a sample of up to 500
    values to keep trigger() O(1) in dataset size.
    """
    if pd.api.types.is_float_dtype(s):
        vals = s.dropna().to_numpy()
        if len(vals) == 0:
            return False
        frac = np.abs(vals - np.round(vals))
        return bool(np.any(frac > 1e-5))

    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_bool_dtype(s):
        return False

    # Filter out missing strings — sample to bound cost on large columns
    mask = s.notna() & ~s.isin(_NA_STRINGS)
    clean = s[mask]
    if len(clean) == 0:
        return False

    # Sample up to 500 values so the pd.to_numeric call stays fast
    _SAMPLE = 500
    if len(clean) > _SAMPLE:
        clean = clean.iloc[:_SAMPLE]

    num = pd.to_numeric(clean, errors="coerce")
    valid_num = num.dropna()
    if len(valid_num) == 0:
        return False

    if len(valid_num) / len(clean) < 0.90:
        return False

    vals = valid_num.to_numpy(dtype=float)
    frac = np.abs(vals - np.round(vals))
    return bool(np.mean(frac > 1e-5) > 0.05)


class LeakageIdentifierHypothesis(Hypothesis):
    """H3b Identifier/row-order leakage hypothesis."""

    type: HypothesisType = HypothesisType.leakage
    name: str = "leakage_identifier"

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        """Applies when training and holdout features exist."""
        if ctx.split.X_train is None or ctx.split.X_holdout is None:
            return Applicability(applicable=False, reason="Missing train or holdout split.")
        return Applicability(applicable=True)

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        """Trigger LK-B-T1: S1 high unique ratio or S2 row-order correlation."""
        thresh = ctx.thresholds
        id_unique_ratio = thresh.id_unique_ratio if thresh else 0.95
        id_rowcorr = thresh.id_rowcorr if thresh else 0.95

        X_tr = ctx.split.X_train
        y_tr = np.asarray(ctx.split.y_train, dtype=float)
        n_tr = len(X_tr)

        flagged_s1: list[str] = []
        flagged_s2: list[str] = []

        # Check S1
        for col in X_tr.columns:
            s = X_tr[col]
            if s.isna().all() or s.nunique(dropna=True) <= 1:
                continue
            if _is_continuous_numeric(s):
                continue
            u_ratio = float(s.nunique(dropna=True) / max(1, n_tr))
            if u_ratio >= id_unique_ratio:
                flagged_s1.append(col)

        # Check S2 (only single_file and split_column modes)
        split_mode = ctx.config.split.mode if ctx.config and ctx.config.split else None
        if split_mode in (SplitMode.single_file, SplitMode.split_column):
            if hasattr(ctx.split, "orig_index_train") and len(ctx.split.orig_index_train) == len(
                y_tr
            ):
                row_idx = np.asarray(ctx.split.orig_index_train, dtype=float)
            else:
                row_idx = np.asarray(X_tr.index, dtype=float)

            res_y = spearmanr(row_idx, y_tr)
            rho_row_y = float(getattr(res_y, "statistic", getattr(res_y, "correlation", 0.0)))
            p_row_y = float(getattr(res_y, "pvalue", 1.0))

            if abs(rho_row_y) >= ID_LABEL_RHO and p_row_y < ID_LABEL_P:
                for col in X_tr.columns:
                    s = X_tr[col]
                    if pd.api.types.is_numeric_dtype(s):
                        if s.isna().any() or s.nunique(dropna=True) <= 1:
                            continue
                        feat_vals = np.asarray(s, dtype=float)
                    else:
                        mask = s.notna() & ~s.isin(_NA_STRINGS)
                        if mask.sum() != len(s):
                            continue
                        s_num = pd.to_numeric(s, errors="coerce")
                        if s_num.isna().any() or s_num.nunique(dropna=True) <= 1:
                            continue
                        feat_vals = s_num.to_numpy(dtype=float)

                    res_f = spearmanr(feat_vals, row_idx)
                    rho_f_row = float(
                        getattr(res_f, "statistic", getattr(res_f, "correlation", 0.0))
                    )
                    if abs(rho_f_row) >= id_rowcorr:
                        flagged_s2.append(col)

        all_flagged = sorted(set(flagged_s1 + flagged_s2))
        fired = len(all_flagged) > 0

        vals: dict[str, float | str | bool | int | None] = {
            "flagged_features": ",".join(all_flagged) if all_flagged else None,
            "n_flagged": len(all_flagged),
            "s1_count": len(flagged_s1),
            "s2_count": len(flagged_s2),
        }
        return TriggerResult(fired=fired, rule_id="LK-B-T1", values=vals)

    def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list[Any]:
        """Returns empty list; secondary ablation runs directly in evaluate()."""
        return []

    def evaluate(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
        results: list[Any],
    ) -> list[HypothesisResult]:
        """Evaluate identifier leakage via permutation importance share and ablation."""
        rule = RULE_LK_B_D1
        profile = ctx.config.rules_profile
        hyp_id = self.name
        caveats = [
            "Identifier-like features can be legitimate; "
            "row-order checks require original order and are skipped in test_file mode."
        ]

        if not trigger.fired:
            decision = Decision(
                rule_id=rule.id,
                rule_text=rule.text,
                rules_profile=profile,
                criteria_met={"flagged_identifier": False, "importance_supported": False},
            )
            return [
                HypothesisResult(
                    id=hyp_id,
                    type=self.type,
                    subtype="identifier",
                    status=Status.not_triggered,
                    evidence_strength=0.0,
                    severity=None,
                    trigger=trigger,
                    decision=decision,
                    caveats=caveats,
                    recommended_actions=[
                        "Confirm identifier features are excluded from feature set."
                    ],
                )
            ]

        # Calculate permutation importance shares
        shares = permutation_importance_share(ctx, n_repeats=IMPORTANCE_REPEATS)

        flagged_str = str(trigger.values.get("flagged_features") or "")
        flagged_features = [f for f in flagged_str.split(",") if f]
        if not flagged_features:
            flagged_features = list(ctx.split.X_holdout.columns)

        # Driver feature: highest importance share; alphabetical tie-break
        driver = sorted(flagged_features, key=lambda f: (-shares.get(f, 0.0), f))[0]
        driver_share = shares.get(driver, 0.0)

        thresh = ctx.thresholds
        id_imp_supp = thresh.id_imp_supported if thresh else 0.25
        id_imp_rej = thresh.id_imp_rejected if thresh else 0.05

        # Structural status (independent of rules profile)
        if driver_share >= id_imp_supp:
            status = Status.supported
        elif driver_share >= id_imp_rej:
            status = Status.weakly_supported
        else:
            status = Status.rejected

        criteria_met = {
            "flagged_identifier": True,
            "importance_supported": bool(driver_share >= id_imp_supp),
        }
        # Strength: both criteria = 1.0, one = 0.5, neither = 0.0
        if criteria_met["flagged_identifier"] and criteria_met["importance_supported"]:
            strength = 1.0
        elif criteria_met["flagged_identifier"] or criteria_met["importance_supported"]:
            strength = 0.5
        else:
            strength = 0.0

        # Secondary evidence: ablation
        ablation_boot = None
        reduced_cols = [c for c in ctx.split.X_train.columns if c != driver]
        reduced_col_types = {c: ctx.col_types[c] for c in reduced_cols if c in ctx.col_types}
        y_ho = np.asarray(ctx.split.y_holdout, dtype=int)
        metric = ctx.config.primary_metric.value
        use_proba = needs_proba(metric)

        p_base = np.asarray(ctx.baseline_run.holdout_pred.y_pred, dtype=int)
        pr_base = (
            np.asarray(ctx.baseline_run.holdout_pred.y_proba, dtype=float) if use_proba else None
        )

        try:
            if ctx.budget_check:
                ctx.budget_check()

            adapter = get_model(ctx.config.model_spec, reduced_col_types, seed=ctx.config.seed)
            adapter.fit(ctx.split.X_train[reduced_cols], ctx.split.y_train)

            p_abl = np.asarray(adapter.predict(ctx.split.X_holdout[reduced_cols]), dtype=int)
            pr_abl = (
                np.asarray(
                    adapter.predict_proba(ctx.split.X_holdout[reduced_cols])[:, 1], dtype=float
                )
                if use_proba
                else None
            )

            ablation_boot = paired_bootstrap(
                y_true=y_ho,
                pred_a=p_abl,
                pred_b=p_base,
                metric=metric,
                n_boot=ctx.thresholds.n_boot if ctx.thresholds else 1000,
                seed=ctx.seed_for("leakage", "ablation"),
                alpha=ctx.thresholds.alpha if ctx.thresholds else 0.05,
                y_proba_a=pr_abl,
                y_proba_b=pr_base,
            )
        except Exception as exc:
            caveats.append(f"Ablation evaluation omitted due to constraint: {exc}")

        # Effect and severity
        effect = None
        if ablation_boot is not None:
            effect = EffectResult(
                metric=metric,
                baseline=float(ablation_boot.metric_a),
                intervention=float(ablation_boot.metric_b),
                delta=float(ablation_boot.delta),
                ci_low=float(ablation_boot.ci_low),
                ci_high=float(ablation_boot.ci_high),
                n=len(y_ho),
            )

        delta_for_sev = ablation_boot.delta if ablation_boot else driver_share
        sev = (
            severity(delta_for_sev, ctx.thresholds)
            if status in (Status.supported, Status.weakly_supported)
            else None
        )

        # Diagnostic evidence
        ev_ids = [
            make_evidence_id(f"importance_share_{driver}", driver_share, "permutation", None),
        ]
        if ablation_boot is not None:
            ev_ids.append(
                make_evidence_id(
                    "ablation_dependence", ablation_boot.delta, "paired_bootstrap", None
                )
            )

        decision = Decision(
            rule_id=rule.id,
            rule_text=rule.text,
            rules_profile=profile,
            criteria_met=criteria_met,
        )

        actions = [
            f"Remove '{driver}' or confirm it is available and meaningful at prediction time."
        ]

        return [
            HypothesisResult(
                id=hyp_id,
                type=self.type,
                subtype="identifier",
                status=status,
                evidence_strength=strength,
                severity=sev,
                trigger=trigger,
                diagnostic_evidence_ids=ev_ids,
                effect=effect,
                decision=decision,
                caveats=caveats,
                recommended_actions=actions,
            )
        ]


# ---------------------------------------------------------------------------
# H3c: Target-Proxy Leakage
# ---------------------------------------------------------------------------

MAX_SUSPECTS: int = 3
MAX_ABLATION_FEATURES: int = 10
GROUP_RHO: float = 0.9
SUSPICIOUS_NAME_PATTERNS: tuple[str, ...] = (
    "result",
    "outcome",
    "target",
    "label",
    "leak",
    "future",
    "after",
)


def single_feature_score(
    feature: str,
    X: pd.DataFrame,
    y: np.ndarray,
    *,
    cv: int = 3,
    seed: int = 0,
) -> float:
    """Evaluate single feature predictive power via depth-4 decision tree and 3-fold CV.

    Out-of-fold probabilities and ROC-AUC.
    """
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.preprocessing import OrdinalEncoder
    from sklearn.tree import DecisionTreeClassifier

    y_arr = np.asarray(y, dtype=int).ravel()
    classes = np.unique(y_arr)
    if len(classes) < 2:
        return 0.5

    s = X[feature]
    if pd.api.types.is_numeric_dtype(s):
        vals = s.astype(float).fillna(0.0).to_numpy()
        feat_mat = vals.reshape(-1, 1)
    else:
        s_clean = s.astype(str).fillna("__missing__").to_numpy().reshape(-1, 1)
        enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
        feat_mat = enc.fit_transform(s_clean)

    n = len(y_arr)
    min_class_count = min(int(np.sum(y_arr == c)) for c in classes)
    effective_cv = min(cv, min_class_count)
    if effective_cv < 2:
        return 0.5

    skf = StratifiedKFold(n_splits=effective_cv, shuffle=True, random_state=seed)
    oof_proba = np.zeros(n, dtype=float)

    for tr_idx, val_idx in skf.split(feat_mat, y_arr):
        tree = DecisionTreeClassifier(max_depth=4, random_state=seed)
        tree.fit(feat_mat[tr_idx], y_arr[tr_idx])
        probs = tree.predict_proba(feat_mat[val_idx])
        if probs.shape[1] > 1:
            oof_proba[val_idx] = probs[:, 1]
        else:
            oof_proba[val_idx] = float(tree.classes_[0] == 1)

    try:
        score = float(roc_auc_score(y_arr, oof_proba))
        return score if math.isfinite(score) else 0.5
    except Exception:
        return 0.5


def flip_effect(boot: BootstrapResult) -> BootstrapResult:
    """Invert effect direction: negate delta, swap/negate CI bounds, swap metric_a/metric_b."""
    return BootstrapResult(
        delta=-float(boot.delta),
        ci_low=-float(boot.ci_high),
        ci_high=-float(boot.ci_low),
        se=float(boot.se),
        n_valid=boot.n_valid,
        n_boot=boot.n_boot,
        metric_a=float(boot.metric_b),
        metric_b=float(boot.metric_a),
    )


def status_for_proxy(
    generic_status: Status,
    suspects: list[str],
    user_flagged: set[str],
    shares: dict[str, float],
    transfer_supported: bool,
    thresholds: Thresholds | None = None,
) -> Status:
    """Precedence rules for target-proxy leakage under BOTH profiles:

    1. Any user-flagged suspect with importance share >= 0.10 -> supported.
    2. Transfer supported -> supported.
    3. Every suspect user-flagged and every share < 0.05 -> rejected.
    4. Otherwise cap status at weakly_supported.
    """
    flagged_imp_supp = thresholds.flagged_imp_supported if thresholds else 0.10
    flagged_imp_min = thresholds.flagged_imp_min if thresholds else 0.05

    # 1. Any user-flagged suspect with share >= 0.10 -> supported
    if any(s in user_flagged and shares.get(s, 0.0) >= flagged_imp_supp for s in suspects):
        return Status.supported

    # 2. Transfer supported -> supported
    if transfer_supported:
        return Status.supported

    # 3. Every suspect user-flagged and every share < 0.05 -> rejected
    if (
        suspects
        and all(s in user_flagged for s in suspects)
        and all(shares.get(s, 0.0) < flagged_imp_min for s in suspects)
    ):
        return Status.rejected

    # 4. Otherwise cap status at weakly_supported
    if generic_status == Status.supported:
        return Status.weakly_supported

    return generic_status


def _find_correlated_groups(df: pd.DataFrame, seed_features: list[str]) -> set[str]:
    """Find connected components of numeric features with absolute Pearson correlation > 0.9."""
    from scipy.stats import pearsonr

    numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]
    adj: dict[str, set[str]] = {c: set() for c in numeric_cols}

    for i in range(len(numeric_cols)):
        col_i = numeric_cols[i]
        s_i = df[col_i]
        for j in range(i + 1, len(numeric_cols)):
            col_j = numeric_cols[j]
            s_j = df[col_j]
            valid_mask = s_i.notna() & s_j.notna()
            if int(np.sum(valid_mask)) >= 30:
                v_i = s_i[valid_mask].to_numpy(dtype=float)
                v_j = s_j[valid_mask].to_numpy(dtype=float)
                if np.std(v_i) > 1e-9 and np.std(v_j) > 1e-9:
                    r, _ = pearsonr(v_i, v_j)
                    if abs(r) > GROUP_RHO:
                        adj[col_i].add(col_j)
                        adj[col_j].add(col_i)

    # BFS from seed features
    result = set(seed_features)
    for start in seed_features:
        if start not in adj:
            continue
        queue = [start]
        visited = {start}
        while queue:
            curr = queue.pop(0)
            result.add(curr)
            for neighbor in adj[curr]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
    return result


class LeakageTargetProxyHypothesis(Hypothesis):
    """H3c Target-proxy leakage hypothesis."""

    type: HypothesisType = HypothesisType.leakage
    name: str = "leakage_target_proxy"

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        """Applies when training and holdout features exist."""
        if ctx.split.X_train is None or ctx.split.X_holdout is None:
            return Applicability(applicable=False, reason="Missing train or holdout split.")
        return Applicability(applicable=True)

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        """Trigger LK-C-T1: suspect features identified by AUC, importance, or user flag."""
        thresh = ctx.thresholds
        sf_auc_thresh = thresh.sf_auc if thresh else 0.90
        proxy_imp_share = thresh.proxy_imp_share if thresh else 0.50

        X_tr = ctx.split.X_train
        y_tr = np.asarray(ctx.split.y_train, dtype=int)

        check_cols = list(X_tr.columns)[:50]
        shares = permutation_importance_share(ctx)
        user_flagged = set(
            ctx.config.metadata.get("unavailable_at_inference", [])
            if ctx.config and isinstance(ctx.config.metadata, dict)
            else (
                getattr(ctx.config.metadata, "unavailable_at_inference", [])
                if ctx.config and ctx.config.metadata
                else []
            )
        )

        sf_aucs: dict[str, float] = {}
        suspects: list[str] = []

        for f in check_cols:
            auc = single_feature_score(
                f, X_tr, y_tr, cv=3, seed=ctx.seed_for("leakage", f, "sf_auc")
            )
            sf_aucs[f] = auc
            imp_share = shares.get(f, 0.0)

            if (auc >= sf_auc_thresh) or (imp_share >= proxy_imp_share) or (f in user_flagged):
                suspects.append(f)

        # Cap at MAX_SUSPECTS: user-flagged first, then AUC descending, alphabetical ties
        sorted_suspects = sorted(
            suspects,
            key=lambda f: (0 if f in user_flagged else 1, -sf_aucs.get(f, 0.0), f),
        )[:MAX_SUSPECTS]

        fired = len(sorted_suspects) > 0

        vals: dict[str, float | str | bool | int | None] = {
            "suspects": ",".join(sorted_suspects) if sorted_suspects else None,
            "n_suspects": len(sorted_suspects),
            "top_sf_auc": _safe_float(max(sf_aucs.values())) if sf_aucs else 0.0,
        }
        return TriggerResult(fired=fired, rule_id="LK-C-T1", values=vals)

    def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list[Any]:
        """Plan E-LK-3 and optional E-LK-4 ablation experiments."""
        from modelroot.schemas.experiment import ExperimentSpec

        if not trigger.fired:
            return []

        suspects_str = str(trigger.values.get("suspects") or "")
        suspects = [s for s in suspects_str.split(",") if s]
        if not suspects:
            return []

        # Find connected components of correlated numeric features
        ablation_set = _find_correlated_groups(ctx.split.X_train, suspects)

        if len(ablation_set) > MAX_ABLATION_FEATURES:
            return []

        features_param = ",".join(sorted(ablation_set))
        metric = ctx.config.primary_metric.value

        specs = [
            ExperimentSpec(
                id="E-LK-3",
                hypothesis_id=self.name,
                name="ablate_proxy",
                intervention="drop_features",
                params={"features": features_param},
                metric=metric,
                eval_on=["holdout"],
                seed=ctx.seed_for("leakage", "ablate"),
                budget_s=ctx.config.budget_s,
                repeats=0,
            )
        ]

        # E-LK-4 on current data if labeled current dataset is provided
        has_current = (
            getattr(ctx.split, "X_current", None) is not None
            and getattr(ctx.split, "y_current", None) is not None
        )
        if has_current:
            specs.append(
                ExperimentSpec(
                    id="E-LK-4",
                    hypothesis_id=self.name,
                    name="ablate_proxy_current",
                    intervention="drop_features",
                    params={"features": features_param, "evaluate_current": True},
                    metric=metric,
                    eval_on=["holdout", "current"],
                    seed=ctx.seed_for("leakage", "ablate_current"),
                    budget_s=ctx.config.budget_s,
                    repeats=0,
                )
            )

        return specs

    def evaluate(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
        results: list[Any],
    ) -> list[HypothesisResult]:
        """Evaluate target-proxy leakage via ablation dependence and transfer diagnostic."""
        from modelroot.evidence.rules import RULE_LK_C_D1

        rule = RULE_LK_C_D1
        profile = ctx.config.rules_profile
        hyp_id = self.name

        verbatim_caveat = (
            "ModelRoot cannot distinguish proxy leakage from a legitimately strong feature "
            "without knowing whether the feature is available at prediction time; "
            "this finding requires domain review."
        )
        caveats = [verbatim_caveat]

        suspects_str = str(trigger.values.get("suspects") or "")
        suspects = [s for s in suspects_str.split(",") if s]

        # Check name patterns on suspects
        for s in suspects:
            s_lower = s.lower()
            if any(pat in s_lower for pat in SUSPICIOUS_NAME_PATTERNS):
                caveats.append(
                    f"Feature '{s}' name matches suspicious target-proxy keyword; "
                    "domain review recommended."
                )

        actions = [
            f"Confirm whether '{', '.join(sorted(suspects))}' is available "
            "when the model makes predictions; "
            "if not, remove it and retrain."
        ]

        if not trigger.fired:
            decision = Decision(
                rule_id=rule.id,
                rule_text=rule.text,
                rules_profile=profile,
                criteria_met={"suspect_selected": False, "confirmed_proxy": False},
            )
            return [
                HypothesisResult(
                    id=hyp_id,
                    type=self.type,
                    subtype="target_proxy",
                    status=Status.not_triggered,
                    evidence_strength=0.0,
                    severity=None,
                    trigger=trigger,
                    decision=decision,
                    caveats=caveats,
                    recommended_actions=actions,
                )
            ]

        # Compute correlated ablation set
        ablation_set = _find_correlated_groups(ctx.split.X_train, suspects)
        if len(ablation_set) > MAX_ABLATION_FEATURES:
            caveats.append(
                f"Ablation set requires more than 10 features ({len(ablation_set)}); "
                "target-proxy evaluation inconclusive."
            )
            decision = Decision(
                rule_id=rule.id,
                rule_text=rule.text,
                rules_profile=profile,
                criteria_met={"suspect_selected": True, "confirmed_proxy": False},
            )
            return [
                HypothesisResult(
                    id=hyp_id,
                    type=self.type,
                    subtype="target_proxy",
                    status=Status.inconclusive,
                    evidence_strength=0.5,
                    severity=None,
                    trigger=trigger,
                    decision=decision,
                    caveats=caveats,
                    recommended_actions=actions,
                )
            ]

        # Find E-LK-3 result
        res_e3 = next(
            (
                r
                for r in results
                if getattr(r, "spec_id", None) == "E-LK-3"
                or getattr(getattr(r, "spec", None), "id", None) == "E-LK-3"
            ),
            None,
        )
        outcome = None
        if res_e3 is not None:
            if hasattr(res_e3, "outcome") and res_e3.outcome is not None:
                outcome = res_e3.outcome
            else:
                from modelroot.experiments.interventions import drop_features_intervention

                ablation_set = _find_correlated_groups(ctx.split.X_train, suspects)
                try:
                    outcome = drop_features_intervention(
                        ctx, {"features": ",".join(sorted(ablation_set))}
                    )
                except Exception as exc:
                    caveats.append(f"Ablation outcome could not be computed: {exc}")

        if outcome is None:
            # If experiment could not run (e.g. detector-only mode)
            decision = Decision(
                rule_id=rule.id,
                rule_text=rule.text,
                rules_profile=profile,
                criteria_met={"suspect_selected": True, "confirmed_proxy": False},
            )
            return [
                HypothesisResult(
                    id=hyp_id,
                    type=self.type,
                    subtype="target_proxy",
                    status=Status.weakly_supported,
                    evidence_strength=0.5,
                    severity=None,
                    trigger=trigger,
                    decision=decision,
                    caveats=caveats,
                    recommended_actions=actions,
                )
            ]
        y_ho = np.asarray(ctx.split.y_holdout, dtype=int)
        metric = ctx.config.primary_metric.value
        use_proba = needs_proba(metric)

        p_base = np.asarray(ctx.baseline_run.holdout_pred.y_pred, dtype=int)
        pr_base = (
            np.asarray(ctx.baseline_run.holdout_pred.y_proba, dtype=float) if use_proba else None
        )

        p_abl = outcome.holdout_pred
        pr_abl = outcome.holdout_proba

        # Paired bootstrap for dependence: baseline score - ablated score
        boot = paired_bootstrap(
            y_true=y_ho,
            pred_a=p_abl,
            pred_b=p_base,
            metric=metric,
            n_boot=ctx.thresholds.n_boot if ctx.thresholds else 1000,
            seed=ctx.seed_for("leakage", "proxy_boot"),
            alpha=ctx.thresholds.alpha if ctx.thresholds else 0.05,
            y_proba_a=pr_abl,
            y_proba_b=pr_base,
        )

        ci_w = float(boot.ci_high - boot.ci_low)
        t = ctx.thresholds
        gate = power_gate(
            n_holdout=len(y_ho),
            n_pos=int(np.sum(y_ho == 1)),
            ci_width=ci_w,
            holdout_min_n=t.holdout_min_n if t else 200,
            holdout_min_pos=t.holdout_min_pos if t else 30,
            ci_width_max=t.ci_width_max if t else 0.10,
        )

        generic_status, criteria_met = decide_intervention_effect(
            delta=boot.delta,
            ci_low=boot.ci_low,
            ci_high=boot.ci_high,
            gate_ok=gate.ok,
            extra_criteria={"suspect_selected": True},
            thresholds=ctx.thresholds,
            profile=profile,
        )

        # Check transfer diagnostic
        transfer_supported = False
        user_flagged = set(
            ctx.config.metadata.get("unavailable_at_inference", [])
            if ctx.config and isinstance(ctx.config.metadata, dict)
            else (
                getattr(ctx.config.metadata, "unavailable_at_inference", [])
                if ctx.config and ctx.config.metadata
                else []
            )
        )
        shares = permutation_importance_share(ctx)

        if (
            getattr(ctx.split, "X_current", None) is not None
            and getattr(ctx.split, "y_current", None) is not None
        ):
            driver = suspects[0]
            sf_auc_base = single_feature_score(
                driver,
                ctx.split.X_train,
                np.asarray(ctx.split.y_train),
                cv=3,
                seed=ctx.seed_for("leakage", driver, "sf_auc"),
            )
            sf_auc_curr = single_feature_score(
                driver,
                ctx.split.X_current,
                np.asarray(ctx.split.y_current),
                cv=3,
                seed=ctx.seed_for("leakage", driver, "sf_curr"),
            )
            sf_auc_drop = sf_auc_base - sf_auc_curr

            fn = metric_fn(metric)
            y_curr = np.asarray(ctx.split.y_current, dtype=int)
            b_curr_p = np.asarray(ctx.baseline_run.adapter.predict(ctx.split.X_current), dtype=int)
            b_curr_pr: np.ndarray = (
                np.asarray(
                    ctx.baseline_run.adapter.predict_proba(ctx.split.X_current)[:, 1],
                    dtype=float,
                )
                if use_proba
                else np.zeros(len(y_curr), dtype=float)
            )
            b_curr_score = float(fn(y_curr, b_curr_p, b_curr_pr))
            b_ho_score = float(boot.metric_b)
            base_drop = b_ho_score - b_curr_score

            abl_curr_proba: np.ndarray = (
                outcome.current_proba
                if outcome.current_proba is not None
                else np.zeros(len(y_curr), dtype=float)
            )
            abl_curr_score = float(fn(y_curr, outcome.current_pred, abl_curr_proba))
            abl_ho_score = float(boot.metric_a)
            ablated_drop = abl_ho_score - abl_curr_score

            delta_min = t.delta_min if t else 0.02
            transfer_auc_drop = t.transfer_auc_drop if t else 0.15

            if (
                (sf_auc_drop >= transfer_auc_drop)
                and (base_drop >= delta_min)
                and (ablated_drop < 0.5 * base_drop)
            ):
                transfer_supported = True

        status = status_for_proxy(
            generic_status=generic_status,
            suspects=suspects,
            user_flagged=user_flagged,
            shares=shares,
            transfer_supported=transfer_supported,
            thresholds=ctx.thresholds,
        )

        criteria_met["confirmed_proxy"] = bool(status == Status.supported)
        strength = evidence_strength(rule, criteria_met)
        sev = (
            severity(boot.delta, ctx.thresholds)
            if status in (Status.supported, Status.weakly_supported)
            else None
        )

        # Final effect: baseline = ablated score, intervention = full baseline score
        # delta = dependence
        effect = EffectResult(
            metric=metric,
            baseline=float(boot.metric_a),
            intervention=float(boot.metric_b),
            delta=float(boot.delta),
            ci_low=float(boot.ci_low),
            ci_high=float(boot.ci_high),
            n=len(y_ho),
        )

        ev_ids = [
            make_evidence_id("ablation_dependence", float(boot.delta), "paired_bootstrap", None),
        ]
        for s in suspects:
            ev_ids.append(
                make_evidence_id(f"importance_share_{s}", shares.get(s, 0.0), "permutation", None)
            )

        decision = Decision(
            rule_id=rule.id,
            rule_text=rule.text,
            rules_profile=profile,
            criteria_met=criteria_met,
        )

        return [
            HypothesisResult(
                id=hyp_id,
                type=self.type,
                subtype="target_proxy",
                status=status,
                evidence_strength=strength,
                severity=sev,
                trigger=trigger,
                diagnostic_evidence_ids=ev_ids,
                effect=effect,
                decision=decision,
                caveats=caveats,
                recommended_actions=actions,
            )
        ]
