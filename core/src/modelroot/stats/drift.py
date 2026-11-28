"""Dataset drift statistics and multidimensional domain classifier (M19).

Provides:
- Univariate drift: Kolmogorov-Smirnov (numeric), Chi-square (categorical),
  Population Stability Index (PSI with epsilon smoothing), Jensen-Shannon divergence,
  and missingness shift.
- Multidimensional domain classifier: shallow HistGradientBoosting with 5-fold CV pooled
  OOF ROC-AUC, label permutation test with add-one Monte Carlo p-value, and feature importances.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OrdinalEncoder


@dataclass
class UnivariateDriftResult:
    """Univariate drift assessment for a single feature."""

    feature: str
    feature_type: str  # "numeric" or "categorical"
    statistic_name: str  # "ks" or "chi2"
    statistic_value: float
    p_value: float
    psi: float
    js_divergence: float | None = None
    missing_rate_ref: float = 0.0
    missing_rate_cur: float = 0.0


@dataclass
class DomainClassifierResult:
    """Outcome of the multidimensional domain classifier evaluation."""

    auc: float
    p_value: float
    permuted_aucs: list[float] = field(default_factory=list)
    feature_importances: dict[str, float] = field(default_factory=dict)


@dataclass
class DriftReport:
    """Comprehensive dataset drift report."""

    domain_classifier: DomainClassifierResult
    univariate: dict[str, UnivariateDriftResult] = field(default_factory=dict)
    top_drifted_features: list[str] = field(default_factory=list)
    n_ref: int = 0
    n_cur: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert report to JSON-serializable dictionary."""
        return {
            "n_reference": self.n_ref,
            "n_current": self.n_cur,
            "domain_classifier": {
                "auc": float(self.domain_classifier.auc),
                "p_value": float(self.domain_classifier.p_value),
                "top_features": self.domain_classifier.feature_importances,
            },
            "top_drifted_features": self.top_drifted_features,
            "univariate_drift": {
                feat: {
                    "feature_type": u.feature_type,
                    "statistic_name": u.statistic_name,
                    "statistic_value": float(u.statistic_value),
                    "p_value": float(u.p_value),
                    "psi": float(u.psi),
                    "js_divergence": (
                        float(u.js_divergence) if u.js_divergence is not None else None
                    ),
                    "missing_rate_ref": float(u.missing_rate_ref),
                    "missing_rate_cur": float(u.missing_rate_cur),
                }
                for feat, u in self.univariate.items()
            },
        }


def compute_univariate_drift(
    ref_s: pd.Series,
    cur_s: pd.Series,
    feature_name: str,
    is_numeric: bool,
    eps: float = 1e-4,
) -> UnivariateDriftResult:
    """Compute univariate drift statistics for a single feature."""
    m_ref = float(ref_s.isna().mean()) if len(ref_s) > 0 else 0.0
    m_cur = float(cur_s.isna().mean()) if len(cur_s) > 0 else 0.0

    ref_valid = ref_s.dropna()
    cur_valid = cur_s.dropna()

    if len(ref_valid) == 0 or len(cur_valid) == 0:
        return UnivariateDriftResult(
            feature=feature_name,
            feature_type="numeric" if is_numeric else "categorical",
            statistic_name="ks" if is_numeric else "chi2",
            statistic_value=0.0,
            p_value=1.0,
            psi=0.0,
            missing_rate_ref=m_ref,
            missing_rate_cur=m_cur,
        )

    if is_numeric:
        # 1. Kolmogorov-Smirnov 2-sample test
        ks_res = stats.ks_2samp(ref_valid, cur_valid)
        stat_val = float(ks_res.statistic)
        p_val = float(ks_res.pvalue)

        # 2. Population Stability Index (PSI) with reference deciles
        ref_arr = ref_valid.to_numpy(dtype=float)
        cur_arr = cur_valid.to_numpy(dtype=float)

        quantiles = np.linspace(0.0, 100.0, 11)
        bin_edges = np.percentile(ref_arr, quantiles)
        # Ensure strictly ascending edges
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) < 2:
            # Fallback to range
            low = float(min(np.min(ref_arr), np.min(cur_arr)))
            high = float(max(np.max(ref_arr), np.max(cur_arr)))
            if low == high:
                bin_edges = np.array([low - 1.0, high + 1.0])
            else:
                bin_edges = np.linspace(low, high, 11)

        # Extend extreme bounds to cover all values
        bin_edges[0] = -np.inf
        bin_edges[-1] = np.inf

        ref_counts, _ = np.histogram(ref_arr, bins=bin_edges)
        cur_counts, _ = np.histogram(cur_arr, bins=bin_edges)

        p = ref_counts.astype(float) / float(len(ref_arr))
        q = cur_counts.astype(float) / float(len(cur_arr))

        # Laplace smoothing
        p_smooth = (p + eps) / np.sum(p + eps)
        q_smooth = (q + eps) / np.sum(q + eps)

        psi_val = float(np.sum((q_smooth - p_smooth) * np.log(q_smooth / p_smooth)))

        return UnivariateDriftResult(
            feature=feature_name,
            feature_type="numeric",
            statistic_name="ks",
            statistic_value=stat_val,
            p_value=p_val,
            psi=max(0.0, psi_val),
            missing_rate_ref=m_ref,
            missing_rate_cur=m_cur,
        )
    else:
        # Categorical feature
        ref_val_counts = ref_valid.value_counts()
        cur_val_counts = cur_valid.value_counts()

        categories = sorted(set(ref_val_counts.index) | set(cur_val_counts.index), key=str)
        n_cat = len(categories)

        if n_cat < 2:
            return UnivariateDriftResult(
                feature=feature_name,
                feature_type="categorical",
                statistic_name="chi2",
                statistic_value=0.0,
                p_value=1.0,
                psi=0.0,
                js_divergence=0.0,
                missing_rate_ref=m_ref,
                missing_rate_cur=m_cur,
            )

        ref_freqs = np.array([ref_val_counts.get(c, 0) for c in categories], dtype=float)
        cur_freqs = np.array([cur_val_counts.get(c, 0) for c in categories], dtype=float)

        # 1. Chi-square test
        table = np.vstack([ref_freqs, cur_freqs])
        try:
            chi2_res = stats.chi2_contingency(table)
            chi2_stat = float(chi2_res.statistic)
            chi2_p = float(chi2_res.pvalue)
        except Exception:
            chi2_stat = 0.0
            chi2_p = 1.0

        # 2. Categorical PSI & Jensen-Shannon Divergence
        p = ref_freqs / float(len(ref_valid))
        q = cur_freqs / float(len(cur_valid))

        p_smooth = (p + eps) / np.sum(p + eps)
        q_smooth = (q + eps) / np.sum(q + eps)

        psi_val = float(np.sum((q_smooth - p_smooth) * np.log(q_smooth / p_smooth)))

        # JS Divergence
        m = 0.5 * (p_smooth + q_smooth)
        js_div = float(
            0.5 * np.sum(p_smooth * np.log(p_smooth / m))
            + 0.5 * np.sum(q_smooth * np.log(q_smooth / m))
        )

        return UnivariateDriftResult(
            feature=feature_name,
            feature_type="categorical",
            statistic_name="chi2",
            statistic_value=chi2_stat,
            p_value=chi2_p,
            psi=max(0.0, psi_val),
            js_divergence=max(0.0, js_div),
            missing_rate_ref=m_ref,
            missing_rate_cur=m_cur,
        )


def compute_domain_classifier(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    seed: int = 0,
    n_permutations: int = 100,
    max_samples_per_domain: int = 5000,
) -> DomainClassifierResult:
    """Evaluate multidimensional domain drift via shallow classifier and permutation test."""
    rng = np.random.default_rng(seed)

    # Subsample if dataset is large to protect execution budget
    n_ref = len(df_ref)
    n_cur = len(df_cur)

    if n_ref > max_samples_per_domain:
        idx_ref = rng.choice(n_ref, size=max_samples_per_domain, replace=False)
        sub_ref = df_ref.iloc[idx_ref].copy()
    else:
        sub_ref = df_ref.copy()

    if n_cur > max_samples_per_domain:
        idx_cur = rng.choice(n_cur, size=max_samples_per_domain, replace=False)
        sub_cur = df_cur.iloc[idx_cur].copy()
    else:
        sub_cur = df_cur.copy()

    X_all = pd.concat([sub_ref, sub_cur], ignore_index=True)
    y_all = np.concatenate([np.zeros(len(sub_ref)), np.ones(len(sub_cur))]).astype(int)

    # Preprocess features: encode strings/categoricals
    X_encoded = X_all.copy()
    cat_cols: list[str] = []
    for col in X_encoded.columns:
        if (
            X_encoded[col].dtype == object
            or isinstance(X_encoded[col].dtype, pd.CategoricalDtype)
            or pd.api.types.is_string_dtype(X_encoded[col])
        ):
            cat_cols.append(col)
            # Use OrdinalEncoder handling unknown categories
            enc = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
            # Fill NAs with missing marker string for encoding
            s = X_encoded[col].astype(str)
            X_encoded[col] = enc.fit_transform(s.to_numpy().reshape(-1, 1)).flatten()
        else:
            X_encoded[col] = pd.to_numeric(X_encoded[col], errors="coerce")

    X_mat = X_encoded.to_numpy(dtype=np.float64)

    # Fit 5-fold CV to get observed pooled out-of-fold AUC
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
    oof_probs = np.zeros(len(y_all), dtype=np.float64)

    for tr_idx, val_idx in skf.split(X_mat, y_all):
        clf = HistGradientBoostingClassifier(
            max_depth=3,
            max_iter=100,
            random_state=seed,
        )
        clf.fit(X_mat[tr_idx], y_all[tr_idx])
        oof_probs[val_idx] = clf.predict_proba(X_mat[val_idx])[:, 1]

    try:
        auc_obs = float(roc_auc_score(y_all, oof_probs))
    except Exception:
        auc_obs = 0.50

    # Permutation test on domain labels
    permuted_aucs: list[float] = []
    extreme_count = 0

    if n_permutations > 0:
        for b in range(n_permutations):
            perm_rng = np.random.default_rng(seed + 1001 + b)
            y_perm = perm_rng.permutation(y_all)

            skf_perm = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)
            oof_perm_probs = np.zeros(len(y_all), dtype=np.float64)

            for tr_idx, val_idx in skf_perm.split(X_mat, y_perm):
                clf_perm = HistGradientBoostingClassifier(
                    max_depth=3,
                    max_iter=100,
                    random_state=seed,
                )
                clf_perm.fit(X_mat[tr_idx], y_perm[tr_idx])
                oof_perm_probs[val_idx] = clf_perm.predict_proba(X_mat[val_idx])[:, 1]

            try:
                auc_b = float(roc_auc_score(y_perm, oof_perm_probs))
            except Exception:
                auc_b = 0.50

            permuted_aucs.append(auc_b)
            if auc_b >= auc_obs:
                extreme_count += 1

        p_value = (extreme_count + 1.0) / (n_permutations + 1.0)
    else:
        p_value = 1.0

    # Calculate feature importances on observed fit using permutation importance
    full_clf = HistGradientBoostingClassifier(
        max_depth=3,
        max_iter=100,
        random_state=seed,
    )
    full_clf.fit(X_mat, y_all)

    try:
        pi = permutation_importance(
            full_clf,
            X_mat,
            y_all,
            n_repeats=5,
            random_state=seed,
            scoring="roc_auc",
        )
        importances = {
            col: float(max(0.0, pi.importances_mean[i])) for i, col in enumerate(X_all.columns)
        }
    except Exception:
        importances = dict.fromkeys(X_all.columns, 0.0)

    # Sort importances descending
    sorted_importances = dict(sorted(importances.items(), key=lambda item: item[1], reverse=True))

    return DomainClassifierResult(
        auc=auc_obs,
        p_value=p_value,
        permuted_aucs=permuted_aucs,
        feature_importances=sorted_importances,
    )


def compute_drift_report(
    df_ref: pd.DataFrame,
    df_cur: pd.DataFrame,
    target_column: str | None = None,
    seed: int = 0,
    n_permutations: int = 50,
) -> DriftReport:
    """Run full drift analysis: univariate tests and multidimensional domain classifier."""
    features_ref = df_ref.copy()
    features_cur = df_cur.copy()

    if target_column:
        if target_column in features_ref.columns:
            features_ref = features_ref.drop(columns=[target_column])
        if target_column in features_cur.columns:
            features_cur = features_cur.drop(columns=[target_column])

    # Shared feature set
    common_cols = [c for c in features_ref.columns if c in features_cur.columns]
    features_ref = features_ref[common_cols]
    features_cur = features_cur[common_cols]

    # 1. Univariate drift
    univariate_results: dict[str, UnivariateDriftResult] = {}
    for col in common_cols:
        is_num = bool(
            pd.api.types.is_numeric_dtype(features_ref[col])
            and not pd.api.types.is_bool_dtype(features_ref[col])
        )
        res = compute_univariate_drift(
            ref_s=features_ref[col],
            cur_s=features_cur[col],
            feature_name=col,
            is_numeric=is_num,
        )
        univariate_results[col] = res

    # 2. Domain classifier
    domain_res = compute_domain_classifier(
        df_ref=features_ref,
        df_cur=features_cur,
        seed=seed,
        n_permutations=n_permutations,
    )

    # Rank features by drift effect (PSI or domain importance)
    # Primary sort: PSI descending
    ranked_by_psi = sorted(
        common_cols,
        key=lambda c: (
            univariate_results[c].psi,
            domain_res.feature_importances.get(c, 0.0),
        ),
        reverse=True,
    )

    return DriftReport(
        domain_classifier=domain_res,
        univariate=univariate_results,
        top_drifted_features=ranked_by_psi,
        n_ref=len(df_ref),
        n_cur=len(df_cur),
    )
