"""Repeated subsampling and consistency calculation (M8).

``repeated_subsample_deltas`` fits two model factories on J stratified
80/20 subsamples of the training workspace and returns the per-repeat deltas
together with the subsample sizes used by the Nadeau-Bengio correction.
``consistency`` counts the fraction of repeats where delta > 0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

from modelroot.assessment.metrics import UndefinedMetric, metric_fn, needs_proba
from modelroot.stats.bootstrap import BootstrapError


@dataclass(frozen=True)
class RepeatedResult:
    """Output of repeated subsampling.

    Attributes:
        deltas: Per-repeat metric deltas (metric_b - metric_a), length J.
        n_train: Number of training rows in each subsample.
        n_test: Number of test rows in each subsample.
    """

    deltas: list[float]
    n_train: int
    n_test: int


def repeated_subsample_deltas(
    make_a: Callable[[], Any],
    make_b: Callable[[], Any],
    X: pd.DataFrame,
    y: pd.Series,
    metric: str,
    J: int = 10,
    test_frac: float = 0.2,
    seed: int = 0,
) -> RepeatedResult:
    """Compute metric deltas over J stratified 80/20 subsamples.

    Each repeat:
      1. Draws a stratified subsample (train / test) of *X* / *y*.
      2. Calls ``make_a()`` and ``make_b()`` to obtain fresh model factories.
      3. Fits both on the subsample training set.
      4. Evaluates the metric on the subsample test set.
      5. Appends ``metric_b - metric_a`` to *deltas*.

    Hyperparameters are selected *before* this function is called, so both
    factories receive fixed parameters (no selection bias from inner CV).

    Args:
        make_a: Zero-argument factory that returns an unfitted adapter A.
        make_b: Zero-argument factory that returns an unfitted adapter B.
        X: Full training-workspace feature DataFrame.
        y: Full training-workspace integer target Series.
        metric: Metric name to compare.
        J: Number of repeats (>= 2).
        test_frac: Fraction of workspace to use as inner test set (0, 1).
        seed: Base RNG seed; repeat *j* uses ``seed + j`` for reproducibility.

    Returns:
        :class:`RepeatedResult` with *deltas*, *n_train*, *n_test*.

    Raises:
        ValueError: If J < 2, test_frac out of (0, 1), or stratification fails.
        BootstrapError: If more than 20 % of repeats are invalid.
    """
    if J < 2:
        raise ValueError(f"J must be >= 2, got {J}")
    if not (0.0 < test_frac < 1.0):
        raise ValueError(f"test_frac must be in (0, 1), got {test_frac}")

    fn = metric_fn(metric)
    use_proba = needs_proba(metric)

    X_arr = X.values if hasattr(X, "values") else np.asarray(X)
    y_arr = np.asarray(y, dtype=int)

    n = len(y_arr)
    n_test = max(1, round(n * test_frac))
    n_train = n - n_test

    if n_test < 1 or n_train < 2:
        raise ValueError(f"Too few samples for repeated subsampling: n={n}, test_frac={test_frac}")

    # Validate class feasibility
    classes, counts = np.unique(y_arr, return_counts=True)
    if len(classes) < 2:
        raise ValueError("y must have at least two classes for stratified splitting.")
    min_count = int(counts.min())
    if min_count < 2:
        raise ValueError(
            f"Stratification requires at least 2 samples per class; minority class has {min_count}."
        )

    deltas: list[float] = []
    n_invalid = 0

    # Use StratifiedShuffleSplit for each repeat with deterministic seeds
    for j in range(J):
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_frac, random_state=seed + j)
        train_idx, test_idx = next(sss.split(X_arr, y_arr))

        X_tr = X.iloc[train_idx] if hasattr(X, "iloc") else pd.DataFrame(X_arr[train_idx])
        y_tr = y.iloc[train_idx] if hasattr(y, "iloc") else pd.Series(y_arr[train_idx])
        X_te = X.iloc[test_idx] if hasattr(X, "iloc") else pd.DataFrame(X_arr[test_idx])
        y_te_arr = y_arr[test_idx]

        adapter_a = make_a()
        adapter_b = make_b()

        try:
            adapter_a.fit(X_tr, y_tr)
            adapter_b.fit(X_tr, y_tr)

            pred_a = np.asarray(adapter_a.predict(X_te))
            pred_b = np.asarray(adapter_b.predict(X_te))

            proba_a = (
                np.asarray(adapter_a.predict_proba(X_te))[:, 1]
                if use_proba
                else np.zeros(len(y_te_arr))
            )
            proba_b = (
                np.asarray(adapter_b.predict_proba(X_te))[:, 1]
                if use_proba
                else np.zeros(len(y_te_arr))
            )

            va = fn(y_te_arr, pred_a, proba_a)
            vb = fn(y_te_arr, pred_b, proba_b)
            deltas.append(vb - va)
        except (UndefinedMetric, ValueError):
            n_invalid += 1

    if n_invalid > 0.2 * J:
        raise BootstrapError(
            f"{n_invalid}/{J} repeats were invalid (> 20 %); "
            "data may be too imbalanced for repeated subsampling."
        )

    if not deltas:
        raise BootstrapError("All repeated subsampling repeats were invalid.")

    actual_n_train = n_train
    actual_n_test = n_test

    return RepeatedResult(
        deltas=deltas,
        n_train=actual_n_train,
        n_test=actual_n_test,
    )


def consistency(deltas: list[float]) -> float:
    """Compute the fraction of deltas that are strictly positive.

    Args:
        deltas: List of metric deltas from repeated subsampling.

    Returns:
        Float in [0, 1]; fraction of deltas > 0.

    Raises:
        ValueError: If *deltas* is empty.
    """
    if not deltas:
        raise ValueError("deltas must not be empty.")
    return float(sum(d > 0 for d in deltas) / len(deltas))
