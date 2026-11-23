"""Oracle impact estimation (M13).

Measures performance degradation between clean control and an injected scenario
on a held-out oracle evaluation set:
    impact = clean_metric - scenario_metric
If impact < 0.02 => no_effect = True (excluded from recall denominator per §6.2).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from modelroot.assessment.metrics import metric_fn, needs_proba
from modelroot.models.registry import get_model
from modelroot.schemas.config import ModelSpec


def _to_model_spec(spec: ModelSpec | dict[str, Any]) -> ModelSpec:
    if isinstance(spec, ModelSpec):
        return spec
    return ModelSpec.model_validate(spec)


def impact_from_results(
    clean_score: float,
    scenario_score: float,
    metric: str = "macro_f1",
    threshold: float = 0.02,
) -> tuple[float, bool]:
    """Compute direction-aware performance impact and no_effect flag.

    For higher-is-better metrics, degradation is clean_score - scenario_score.
    For lower-is-better metrics (loss, error, mse, etc.), degradation is
    scenario_score - clean_score.

    Args:
        clean_score: Performance on clean baseline.
        scenario_score: Performance on injected scenario.
        metric: Evaluation metric name.
        threshold: Threshold below which no_effect is True (default 0.02).

    Returns:
        Tuple of (impact_float, no_effect_bool).
    """
    lower_is_better = any(kw in metric.lower() for kw in ("loss", "error", "mse", "mae", "brier"))
    impact = scenario_score - clean_score if lower_is_better else clean_score - scenario_score

    no_effect = impact < threshold
    return float(impact), bool(no_effect)


def oracle_impact(
    clean_df: pd.DataFrame,
    scenario_df: pd.DataFrame,
    target: str,
    clean_model_spec: dict[str, Any] | None = None,
    scenario_model_spec: dict[str, Any] | None = None,
    metric: str = "macro_f1",
    seed: int = 42,
) -> tuple[float, bool]:
    """Compute performance impact and no_effect flag between clean and injected scenarios.

    Args:
        clean_df: Uncorrupted reference DataFrame.
        scenario_df: Injected scenario DataFrame.
        target: Target column name.
        clean_model_spec: Model spec for the clean control.
        scenario_model_spec: Model spec for the scenario.
        metric: Evaluation metric name (default 'macro_f1').
        seed: Random seed.

    Returns:
        Tuple of (impact_float, no_effect_bool).
    """
    clean_spec_raw = clean_model_spec or {
        "algorithm": "random_forest",
        "params": {"n_estimators": 50, "min_samples_leaf": 5, "max_depth": 6},
    }
    scen_spec_raw = scenario_model_spec or clean_spec_raw
    clean_spec = _to_model_spec(clean_spec_raw)
    scen_spec = _to_model_spec(scen_spec_raw)

    fn = metric_fn(metric)
    use_proba = needs_proba(metric)

    # 1. Fit clean baseline on clean data
    # Partition clean_df 80/20 train/oracle-test
    rng = np.random.default_rng(seed)
    n_clean = len(clean_df)
    n_test = max(50, round(n_clean * 0.2))
    idx = np.arange(n_clean)
    rng.shuffle(idx)

    test_idx = idx[:n_test]
    train_idx = idx[n_test:]

    feature_cols_clean = [c for c in clean_df.columns if c not in (target, "split")]
    col_types_clean = dict.fromkeys(feature_cols_clean, "numeric")

    X_clean_tr = clean_df.iloc[train_idx][feature_cols_clean]
    y_clean_tr = clean_df.iloc[train_idx][target]
    X_oracle = clean_df.iloc[test_idx][feature_cols_clean]
    y_oracle = np.asarray(clean_df.iloc[test_idx][target], dtype=int)

    adapter_clean = get_model(clean_spec, col_types_clean, seed=seed)
    adapter_clean.fit(X_clean_tr, y_clean_tr)

    pred_clean = np.asarray(adapter_clean.predict(X_oracle))
    proba_clean = (
        np.asarray(adapter_clean.predict_proba(X_oracle))[:, 1]
        if use_proba
        else np.zeros(len(y_oracle))
    )
    score_clean = float(fn(y_oracle, pred_clean, proba_clean))

    # 2. Fit scenario model on scenario train split
    feature_cols_scen = [c for c in scenario_df.columns if c not in (target, "split")]
    col_types_scen = dict.fromkeys(feature_cols_scen, "numeric")

    if "split" in scenario_df.columns:
        train_mask = scenario_df["split"] == "train"
        test_mask = scenario_df["split"] == "test"
        X_scen_tr = scenario_df[train_mask][feature_cols_scen]
        y_scen_tr = scenario_df[train_mask][target]

        X_scen_te = scenario_df[test_mask][feature_cols_scen]
        y_scen_te = np.asarray(scenario_df[test_mask][target], dtype=int)
    else:
        n_scen = len(scenario_df)
        n_scen_test = max(50, round(n_scen * 0.2))
        scen_idx = np.arange(n_scen)
        rng.shuffle(scen_idx)
        X_scen_tr = scenario_df.iloc[scen_idx[n_scen_test:]][feature_cols_scen]
        y_scen_tr = scenario_df.iloc[scen_idx[n_scen_test:]][target]
        X_scen_te = scenario_df.iloc[scen_idx[:n_scen_test]][feature_cols_scen]
        y_scen_te = np.asarray(scenario_df.iloc[scen_idx[:n_scen_test]][target], dtype=int)

    adapter_scen = get_model(scen_spec, col_types_scen, seed=seed)
    adapter_scen.fit(X_scen_tr, y_scen_tr)

    pred_scen = np.asarray(adapter_scen.predict(X_scen_te))
    proba_scen = (
        np.asarray(adapter_scen.predict_proba(X_scen_te))[:, 1]
        if use_proba
        else np.zeros(len(y_scen_te))
    )
    score_scen = float(fn(y_scen_te, pred_scen, proba_scen))

    return impact_from_results(score_clean, score_scen, metric=metric)
