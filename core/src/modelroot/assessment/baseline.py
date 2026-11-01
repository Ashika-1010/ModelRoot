"""Baseline model fitting and dummy baseline (M5).

``fit_baseline`` trains a fresh clone of the user's adapter on the training
split and stores per-row predictions for all available splits.
``dummy_baseline`` predicts the training majority class with constant scores.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from modelroot.assessment.metrics import _METRIC_REGISTRY, UndefinedMetric, _validate_inputs
from modelroot.data.split import SplitData
from modelroot.models.adapter import ModelAdapter
from modelroot.schemas.metrics import MetricResult

# ---------------------------------------------------------------------------
# PredictionSet
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PredictionSet:
    """Per-row predictions for a single data split.

    Attributes:
        index: Original DataFrame index for these rows.
        y_true: Ground-truth integer labels (0/1).
        y_pred: Predicted integer labels (0/1).
        y_proba: P(class = 1) probability scores.
    """

    index: pd.Index
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: np.ndarray


# ---------------------------------------------------------------------------
# BaselineRun
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BaselineRun:
    """Result of fitting the baseline model on the training split.

    Attributes:
        adapter: The fitted adapter (a clone of the original, not mutated).
        train_pred: Predictions on the training split.
        holdout_pred: Predictions on the holdout split.
        current_pred: Predictions on the current split, if labeled; else None.
        metrics: Metric results for all splits.
        notes: Non-fatal messages (e.g. skipped metrics).
    """

    adapter: ModelAdapter
    train_pred: PredictionSet
    holdout_pred: PredictionSet
    current_pred: PredictionSet | None
    metrics: list[MetricResult]
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# fit_baseline
# ---------------------------------------------------------------------------


def fit_baseline(
    adapter: ModelAdapter,
    split: SplitData,
    primary_metric: str = "macro_f1",
) -> BaselineRun:
    """Fit a clone of *adapter* and record predictions for all splits.

    The supplied *adapter* is never mutated; a fresh clone is fitted.

    Args:
        adapter: Unfitted adapter to clone and train.
        split: Train/holdout/current split data.
        primary_metric: Metric name used for primary reporting; not used here
            but kept for symmetry with reference panel.

    Returns:
        A :class:`BaselineRun` with fitted adapter and predictions.

    Raises:
        ValueError: If prediction output shapes are invalid.
    """
    fitted = adapter.clone_with()
    fitted.fit(split.X_train, split.y_train)

    def _predict(X: pd.DataFrame, y: pd.Series, orig_idx: pd.Index) -> PredictionSet:
        raw_pred = fitted.predict(X)
        raw_proba_2d = fitted.predict_proba(X)
        if raw_proba_2d.ndim != 2 or raw_proba_2d.shape[1] != 2:
            raise ValueError(
                f"predict_proba must return (n, 2) array; got shape {raw_proba_2d.shape}"
            )
        y_proba = raw_proba_2d[:, 1]
        return PredictionSet(
            index=orig_idx,
            y_true=np.asarray(y.values, dtype=int),
            y_pred=np.asarray(raw_pred, dtype=int),
            y_proba=y_proba.astype(float),
        )

    train_pred = _predict(split.X_train, split.y_train, split.orig_index_train)
    holdout_pred = _predict(split.X_holdout, split.y_holdout, split.orig_index_holdout)

    current_pred: PredictionSet | None = None
    if split.X_current is not None and split.y_current is not None:
        current_idx = (
            split.X_current.index
            if hasattr(split.X_current, "index")
            else pd.RangeIndex(len(split.X_current))
        )
        current_pred = _predict(split.X_current, split.y_current, current_idx)

    metrics: list[MetricResult] = []
    notes: list[str] = []

    def _add_metrics(pset: PredictionSet, split_label: str) -> None:
        y_true, y_pred, y_proba = _validate_inputs(pset.y_true, pset.y_pred, pset.y_proba)
        for name, fn in _METRIC_REGISTRY.items():
            try:
                val = fn(y_true, y_pred, y_proba)
                metrics.append(
                    MetricResult(
                        name=name,
                        value=val,
                        split=split_label,  # type: ignore[arg-type]
                        n=len(y_true),
                    )
                )
            except UndefinedMetric as exc:
                notes.append(f"Skipped {name} on {split_label}: {exc}")

    _add_metrics(train_pred, "train")
    _add_metrics(holdout_pred, "holdout")
    if current_pred is not None:
        _add_metrics(current_pred, "current")

    return BaselineRun(
        adapter=fitted,
        train_pred=train_pred,
        holdout_pred=holdout_pred,
        current_pred=current_pred,
        metrics=metrics,
        notes=notes,
    )


# ---------------------------------------------------------------------------
# dummy_baseline
# ---------------------------------------------------------------------------


def dummy_baseline(split: SplitData) -> list[MetricResult]:
    """Compute metrics for a majority-class dummy predictor.

    The dummy predicts the training majority class for every sample.
    Ties (exactly 50/50) are broken toward class 0.
    The constant probability score equals the training positive-class
    prevalence (fraction of training labels == 1).

    Args:
        split: Train/holdout split data.

    Returns:
        List of :class:`MetricResult` for the holdout split.
    """
    y_train = np.asarray(split.y_train.values, dtype=int)
    pos_count = int(np.sum(y_train == 1))
    n_train = len(y_train)
    # Majority class; tie (50%) goes to 0
    majority_class = 1 if pos_count > n_train - pos_count else 0
    prevalence = float(pos_count / n_train) if n_train > 0 else 0.0

    y_holdout = np.asarray(split.y_holdout.values, dtype=int)
    n_holdout = len(y_holdout)
    y_pred = np.full(n_holdout, majority_class, dtype=int)
    y_proba = np.full(n_holdout, prevalence, dtype=float)

    metrics: list[MetricResult] = []
    _, y_pred_v, y_proba_v = _validate_inputs(y_holdout, y_pred, y_proba)

    for name, fn in _METRIC_REGISTRY.items():
        try:
            val = fn(y_holdout, y_pred_v, y_proba_v)
            metrics.append(MetricResult(name=name, value=val, split="holdout", n=n_holdout))
        except UndefinedMetric:
            pass  # Skip undefined metrics silently for dummy

    return metrics
