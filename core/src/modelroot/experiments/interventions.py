"""Intervention registry and outcome type (M9/M11).

Interventions are pure functions registered by name.  The runner looks up
the callable by name from ``ExperimentSpec.intervention``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, TypeVar

import numpy as np
import pandas as pd

InterventionFn = Callable[[Any, dict[str, Any]], Any]
F = TypeVar("F", bound=Callable[[Any, dict[str, Any]], Any])

_INTERVENTIONS: dict[str, InterventionFn] = {}


@dataclass
class InterventionOutcome:
    """Result returned by an intervention function.

    Attributes:
        holdout_pred: Predicted labels on the holdout set (1-D int array).
        holdout_proba: P(class=1) scores on the holdout set (1-D float array).
        fitted_adapter: The fitted model adapter after the intervention.
        factory: Zero-arg callable that returns a fresh *unfitted* adapter
            with the selected hyperparameters fixed (used for repeated subsampling).
        current_pred: Optional predictions on the current set.
        current_proba: Optional P(class=1) scores on the current set.
        notes: Free-text notes (selected params, scores, etc.).
    """

    holdout_pred: np.ndarray
    holdout_proba: np.ndarray
    fitted_adapter: Any
    factory: Callable[[], Any] | None = None
    current_pred: np.ndarray | None = None
    current_proba: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def intervention(name: str) -> Callable[[F], F]:
    """Decorator to register an intervention function by *name*.

    Args:
        name: Stable string identifier for this intervention.

    Returns:
        Decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If *name* is already registered.
    """

    def decorator(fn: F) -> F:
        if name in _INTERVENTIONS:
            raise ValueError(f"Intervention {name!r} is already registered.")
        _INTERVENTIONS[name] = fn
        return fn

    return decorator


def get_intervention(name: str) -> InterventionFn:
    """Look up a registered intervention function.

    Args:
        name: Intervention identifier.

    Returns:
        The registered callable.

    Raises:
        KeyError: If *name* is not registered.
    """
    try:
        return _INTERVENTIONS[name]
    except KeyError:
        raise KeyError(
            f"Intervention {name!r} is not registered. Available: {sorted(_INTERVENTIONS)}"
        ) from None


def list_interventions() -> list[str]:
    """Return all registered intervention names in sorted order."""
    return sorted(_INTERVENTIONS)


# ---------------------------------------------------------------------------
# H2 Class Imbalance Interventions (M14)
# ---------------------------------------------------------------------------


def _get_train_holdout_data(
    ctx: Any,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, np.ndarray, list[str]]:
    """Helper to extract train and holdout features and targets from context."""
    split = ctx.split
    X_tr = split.X_train
    y_tr = pd.Series(split.y_train, dtype=int)
    X_ho = split.X_holdout
    y_ho = np.asarray(split.y_holdout, dtype=int)
    feature_cols = list(X_tr.columns)
    return X_tr, y_tr, X_ho, y_ho, feature_cols


@intervention("class_weight")
def class_weight_intervention(ctx: Any, params: dict[str, Any]) -> InterventionOutcome:
    """E-IM-1: Class weight balancing intervention."""
    from modelroot.models.registry import get_model
    from modelroot.schemas.common import Algorithm
    from modelroot.schemas.config import ModelSpec

    X_tr, y_tr, X_ho, _y_ho, _feature_cols = _get_train_holdout_data(ctx)
    algo = ctx.config.model_spec.algorithm

    if algo == Algorithm.hist_gradient_boosting:
        n_total = len(y_tr)
        n_0 = int(np.sum(y_tr == 0))
        n_1 = int(np.sum(y_tr == 1))
        w_0 = n_total / (2.0 * max(1, n_0))
        w_1 = n_total / (2.0 * max(1, n_1))
        sw = np.where(y_tr == 1, w_1, w_0)

        adapter = get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
        adapter.fit(X_tr, y_tr, sample_weight=sw)

        def factory() -> Any:
            return get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
    else:
        new_params = dict(ctx.config.model_spec.params)
        new_params["class_weight"] = "balanced"
        new_spec = ModelSpec(algorithm=algo, params=new_params)
        adapter = get_model(new_spec, ctx.col_types, seed=ctx.config.seed)
        adapter.fit(X_tr, y_tr)

        def factory() -> Any:
            return get_model(new_spec, ctx.col_types, seed=ctx.config.seed)

    holdout_pred = np.asarray(adapter.predict(X_ho), dtype=int)
    holdout_proba = np.asarray(adapter.predict_proba(X_ho)[:, 1], dtype=float)

    return InterventionOutcome(
        holdout_pred=holdout_pred,
        holdout_proba=holdout_proba,
        fitted_adapter=adapter,
        factory=factory,
        notes=["intervention=class_weight"],
    )


@intervention("threshold_tune")
def threshold_tune_intervention(ctx: Any, params: dict[str, Any]) -> InterventionOutcome:
    """E-IM-2: Classification threshold tuning on inner-CV OOF predictions."""
    from sklearn.model_selection import StratifiedKFold

    from modelroot.assessment.metrics import metric_fn, needs_proba
    from modelroot.models.registry import get_model

    X_tr, y_tr, _X_ho, _y_ho, _feature_cols = _get_train_holdout_data(ctx)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=ctx.config.seed)
    oof_proba = np.zeros(len(y_tr), dtype=float)

    for train_idx, val_idx in skf.split(X_tr, y_tr):
        fold_adapter = get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
        fold_adapter.fit(X_tr.iloc[train_idx], y_tr.iloc[train_idx])
        oof_proba[val_idx] = fold_adapter.predict_proba(X_tr.iloc[val_idx])[:, 1]

    metric_name = ctx.config.primary_metric.value
    fn = metric_fn(metric_name)
    use_pr = needs_proba(metric_name)  # noqa: F841  # kept for readability

    candidates = np.linspace(0.05, 0.95, 91)
    best_score = -float("inf")
    best_tau = 0.5
    y_tr_arr = np.asarray(y_tr, dtype=int)

    for tau in candidates:
        pred_tau = (oof_proba >= tau).astype(int)
        score = float(fn(y_tr_arr, pred_tau, oof_proba))
        if score > best_score + 1e-9:
            best_score = score
            best_tau = float(tau)
        elif abs(score - best_score) <= 1e-9:
            if abs(tau - 0.5) < abs(best_tau - 0.5):
                best_tau = float(tau)

    base_ho_proba = np.asarray(ctx.baseline_run.holdout_pred.y_proba, dtype=float)
    holdout_pred = (base_ho_proba >= best_tau).astype(int)
    holdout_proba = base_ho_proba

    def factory() -> Any:
        return ctx.baseline_run.adapter.clone_with()

    return InterventionOutcome(
        holdout_pred=holdout_pred,
        holdout_proba=holdout_proba,
        fitted_adapter=ctx.baseline_run.adapter,
        factory=factory,
        notes=[
            "intervention=threshold_tune",
            f"selected_threshold={best_tau:.4f}",
            f"oof_metric={best_score:.4f}",
        ],
    )


@intervention("oversample_train")
def oversample_train_intervention(ctx: Any, params: dict[str, Any]) -> InterventionOutcome:
    """E-IM-3: Random oversampling of minority class inside training set only."""
    from modelroot.models.registry import get_model

    X_tr, y_tr, X_ho, _y_ho, _feature_cols = _get_train_holdout_data(ctx)

    n_0 = int(np.sum(y_tr == 0))
    n_1 = int(np.sum(y_tr == 1))
    minority_class = 1 if n_1 < n_0 else 0
    minority_mask = y_tr == minority_class
    majority_mask = ~minority_mask

    min_indices = X_tr[minority_mask].index.to_numpy()
    maj_indices = X_tr[majority_mask].index.to_numpy()

    n_diff = len(maj_indices) - len(min_indices)
    rng = np.random.default_rng(ctx.config.seed)

    if n_diff > 0:
        sampled_indices = rng.choice(min_indices, size=n_diff, replace=True)
        # Invariant: Holdout indices are NEVER sampled
        assert all(idx in X_tr.index for idx in sampled_indices)
        assert not any(idx in X_ho.index for idx in sampled_indices)

        X_tr_ovs = pd.concat([X_tr, X_tr.loc[sampled_indices]], axis=0)
        y_tr_ovs = pd.concat([y_tr, y_tr.loc[sampled_indices]], axis=0)
    else:
        X_tr_ovs, y_tr_ovs = X_tr, y_tr

    adapter = get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
    adapter.fit(X_tr_ovs, y_tr_ovs)

    holdout_pred = np.asarray(adapter.predict(X_ho), dtype=int)
    holdout_proba = np.asarray(adapter.predict_proba(X_ho)[:, 1], dtype=float)

    def factory() -> Any:
        return get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)

    return InterventionOutcome(
        holdout_pred=holdout_pred,
        holdout_proba=holdout_proba,
        fitted_adapter=adapter,
        factory=factory,
        notes=["intervention=oversample_train"],
    )


@intervention("imbalance_best")
def imbalance_best_intervention(ctx: Any, params: dict[str, Any]) -> InterventionOutcome:
    """Inner-CV selection among class_weight, threshold_tune, and oversample_train."""
    from sklearn.model_selection import StratifiedKFold

    from modelroot.assessment.metrics import metric_fn, needs_proba
    from modelroot.models.registry import get_model
    from modelroot.schemas.common import Algorithm
    from modelroot.schemas.config import ModelSpec

    X_tr, y_tr, _X_ho, y_ho, _feature_cols = _get_train_holdout_data(ctx)
    algo = ctx.config.model_spec.algorithm
    metric_name = ctx.config.primary_metric.value
    fn = metric_fn(metric_name)
    use_pr = needs_proba(metric_name)  # noqa: F841  # kept for readability
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=ctx.config.seed)

    oof_cw_pred = np.zeros(len(y_tr), dtype=int)
    oof_cw_proba = np.zeros(len(y_tr), dtype=float)

    oof_base_proba = np.zeros(len(y_tr), dtype=float)

    oof_ovs_pred = np.zeros(len(y_tr), dtype=int)
    oof_ovs_proba = np.zeros(len(y_tr), dtype=float)

    rng = np.random.default_rng(ctx.config.seed)

    for train_idx, val_idx in skf.split(X_tr, y_tr):
        X_f_tr, y_f_tr = X_tr.iloc[train_idx], y_tr.iloc[train_idx]
        X_f_val, _y_f_val = X_tr.iloc[val_idx], y_tr.iloc[val_idx]

        # 1. class_weight
        if algo == Algorithm.hist_gradient_boosting:
            n_t = len(y_f_tr)
            n0 = int(np.sum(y_f_tr == 0))
            n1 = int(np.sum(y_f_tr == 1))
            w0 = n_t / (2.0 * max(1, n0))
            w1 = n_t / (2.0 * max(1, n1))
            sw = np.where(y_f_tr == 1, w1, w0)
            cw_adapter = get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
            cw_adapter.fit(X_f_tr, y_f_tr, sample_weight=sw)
        else:
            p_cw = dict(ctx.config.model_spec.params)
            p_cw["class_weight"] = "balanced"
            cw_spec = ModelSpec(algorithm=algo, params=p_cw)
            cw_adapter = get_model(cw_spec, ctx.col_types, seed=ctx.config.seed)
            cw_adapter.fit(X_f_tr, y_f_tr)

        oof_cw_pred[val_idx] = cw_adapter.predict(X_f_val)
        oof_cw_proba[val_idx] = cw_adapter.predict_proba(X_f_val)[:, 1]

        # 2. baseline for threshold_tune
        base_fold_adapter = get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
        base_fold_adapter.fit(X_f_tr, y_f_tr)
        oof_base_proba[val_idx] = base_fold_adapter.predict_proba(X_f_val)[:, 1]

        # 3. oversample_train
        n0 = int(np.sum(y_f_tr == 0))
        n1 = int(np.sum(y_f_tr == 1))
        m_cls = 1 if n1 < n0 else 0
        min_mask = y_f_tr == m_cls
        maj_mask = ~min_mask
        min_idx = X_f_tr[min_mask].index.to_numpy()
        maj_idx = X_f_tr[maj_mask].index.to_numpy()
        ndiff = len(maj_idx) - len(min_idx)
        if ndiff > 0:
            s_idx = rng.choice(min_idx, size=ndiff, replace=True)
            X_f_ovs = pd.concat([X_f_tr, X_f_tr.loc[s_idx]], axis=0)
            y_f_ovs = pd.concat([y_f_tr, y_f_tr.loc[s_idx]], axis=0)
        else:
            X_f_ovs, y_f_ovs = X_f_tr, y_f_tr

        ovs_adapter = get_model(ctx.config.model_spec, ctx.col_types, seed=ctx.config.seed)
        ovs_adapter.fit(X_f_ovs, y_f_ovs)
        oof_ovs_pred[val_idx] = ovs_adapter.predict(X_f_val)
        oof_ovs_proba[val_idx] = ovs_adapter.predict_proba(X_f_val)[:, 1]

    # Evaluate OOF scores
    y_tr_arr = np.asarray(y_tr, dtype=int)
    score_cw = float(fn(y_tr_arr, oof_cw_pred, oof_cw_proba))
    score_ovs = float(fn(y_tr_arr, oof_ovs_pred, oof_ovs_proba))

    # For threshold_tune: tune best_tau on oof_base_proba
    candidates = np.linspace(0.05, 0.95, 91)
    best_score_tt = -float("inf")
    best_tau = 0.5
    for tau in candidates:
        pred_tau = (oof_base_proba >= tau).astype(int)
        score = float(fn(y_tr_arr, pred_tau, oof_base_proba))
        if score > best_score_tt + 1e-9:
            best_score_tt = score
            best_tau = float(tau)
        elif abs(score - best_score_tt) <= 1e-9:
            if abs(tau - 0.5) < abs(best_tau - 0.5):
                best_tau = float(tau)

    scores = {
        "class_weight": score_cw,
        "threshold_tune": best_score_tt,
        "oversample_train": score_ovs,
    }
    candidates_order = ["class_weight", "threshold_tune", "oversample_train"]
    best_candidate = max(candidates_order, key=lambda c: scores[c])

    # Apply the winning intervention
    if best_candidate == "class_weight":
        outcome = class_weight_intervention(ctx, params)
    elif best_candidate == "threshold_tune":
        outcome = threshold_tune_intervention(ctx, params)
    else:
        outcome = oversample_train_intervention(ctx, params)

    # Calculate ROC-AUC nuance flag
    auc_fn = metric_fn("roc_auc")
    base_auc = float(
        auc_fn(y_ho, ctx.baseline_run.holdout_pred.y_pred, ctx.baseline_run.holdout_pred.y_proba)
    )
    winner_auc = float(auc_fn(y_ho, outcome.holdout_pred, outcome.holdout_proba))
    delta_auc = winner_auc - base_auc
    threshold_prior_issue = abs(delta_auc) <= 0.01

    outcome.notes.append(f"selected_intervention={best_candidate}")
    outcome.notes.append(f"inner_cv_scores={json.dumps(scores)}")
    outcome.notes.append(f"delta_auc={delta_auc:+.4f}")
    outcome.notes.append(f"threshold_prior_issue={threshold_prior_issue}")

    return outcome


@intervention("drop_features")
def drop_features_intervention(ctx: Any, params: dict[str, Any]) -> InterventionOutcome:
    """E-LK-3 / E-LK-4: Drop specified feature subset and fit reduced model (M16)."""
    from modelroot.models.registry import get_model

    if ctx.budget_check:
        ctx.budget_check()

    raw_feats = params.get("features", "")
    if isinstance(raw_feats, str):
        features_to_drop = [f.strip() for f in raw_feats.split(",") if f.strip()]
    elif isinstance(raw_feats, list):
        features_to_drop = list(raw_feats)
    else:
        features_to_drop = []

    X_tr = ctx.split.X_train
    y_tr = ctx.split.y_train
    X_ho = ctx.split.X_holdout

    unknown = [f for f in features_to_drop if f not in X_tr.columns]
    if unknown:
        raise ValueError(f"Unknown features to drop: {unknown}")

    reduced_cols = [c for c in X_tr.columns if c not in features_to_drop]
    reduced_col_types = {c: ctx.col_types[c] for c in reduced_cols if c in ctx.col_types}

    adapter = get_model(ctx.config.model_spec, reduced_col_types, seed=ctx.config.seed)
    adapter.fit(X_tr[reduced_cols], y_tr)

    holdout_pred = np.asarray(adapter.predict(X_ho[reduced_cols]), dtype=int)
    holdout_proba = np.asarray(adapter.predict_proba(X_ho[reduced_cols])[:, 1], dtype=float)

    current_pred = None
    current_proba = None
    if getattr(ctx.split, "X_current", None) is not None:
        X_curr = ctx.split.X_current
        current_pred = np.asarray(adapter.predict(X_curr[reduced_cols]), dtype=int)
        current_proba = np.asarray(adapter.predict_proba(X_curr[reduced_cols])[:, 1], dtype=float)

    return InterventionOutcome(
        holdout_pred=holdout_pred,
        holdout_proba=holdout_proba,
        fitted_adapter=adapter,
        factory=None,
        current_pred=current_pred,
        current_proba=current_proba,
        notes=[f"dropped_features={sorted(features_to_drop)}"],
    )
