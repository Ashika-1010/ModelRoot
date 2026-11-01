"""Metric computation for ModelRoot (M5).

Eight classification metrics are supported, always evaluated with positive
class = 1.  Probability metrics (roc_auc, pr_auc) require a 1-D probability
vector P(class = 1).
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

#: All supported metric names in canonical order.
METRIC_NAMES: tuple[str, ...] = (
    "accuracy",
    "balanced_accuracy",
    "macro_f1",
    "f1",
    "precision",
    "recall",
    "roc_auc",
    "pr_auc",
)

#: Metrics that only need integer predictions.
LABEL_METRICS: frozenset[str] = frozenset(
    {"accuracy", "balanced_accuracy", "macro_f1", "f1", "precision", "recall"}
)

#: Metrics that require P(class = 1) probability scores.
PROBA_METRICS: frozenset[str] = frozenset({"roc_auc", "pr_auc"})


# ---------------------------------------------------------------------------
# Exception
# ---------------------------------------------------------------------------


class UndefinedMetric(Exception):
    """Raised when a metric cannot be computed for the given data.

    Typical causes: only one class present in ``y_true`` when AUC metrics
    are requested, or an unknown metric name.
    """


# ---------------------------------------------------------------------------
# Registry helpers
# ---------------------------------------------------------------------------


def _sklearn_accuracy(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    return float(accuracy_score(y_true, y_pred))


def _sklearn_balanced_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray
) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise UndefinedMetric("balanced_accuracy requires at least two classes.")
    return float(balanced_accuracy_score(y_true, y_pred))


def _sklearn_macro_f1(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise UndefinedMetric("macro_f1 requires at least two classes.")
    return float(f1_score(y_true, y_pred, average="macro", zero_division=0))


def _sklearn_f1(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise UndefinedMetric("f1 requires at least two classes.")
    return float(f1_score(y_true, y_pred, average="binary", pos_label=1, zero_division=0))


def _sklearn_precision(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise UndefinedMetric("precision requires at least two classes.")
    return float(precision_score(y_true, y_pred, pos_label=1, zero_division=0))


def _sklearn_recall(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise UndefinedMetric("recall requires at least two classes.")
    return float(recall_score(y_true, y_pred, pos_label=1, zero_division=0))


def _sklearn_roc_auc(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise UndefinedMetric("roc_auc is undefined when only one class is present.")
    return float(roc_auc_score(y_true, y_proba))


def _sklearn_pr_auc(y_true: np.ndarray, y_pred: np.ndarray, y_proba: np.ndarray) -> float:
    classes = np.unique(y_true)
    if len(classes) < 2:
        raise UndefinedMetric("pr_auc is undefined when only one class is present.")
    return float(average_precision_score(y_true, y_proba))


# Internal registry: name -> callable(y_true, y_pred, y_proba) -> float
_METRIC_REGISTRY: dict[str, Callable[[np.ndarray, np.ndarray, np.ndarray], float]] = {
    "accuracy": _sklearn_accuracy,
    "balanced_accuracy": _sklearn_balanced_accuracy,
    "macro_f1": _sklearn_macro_f1,
    "f1": _sklearn_f1,
    "precision": _sklearn_precision,
    "recall": _sklearn_recall,
    "roc_auc": _sklearn_roc_auc,
    "pr_auc": _sklearn_pr_auc,
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def metric_fn(name: str) -> Callable[[np.ndarray, np.ndarray, np.ndarray], float]:
    """Return the callable for *name*.

    Args:
        name: Metric identifier (must be in ``METRIC_NAMES``).

    Returns:
        A function ``(y_true, y_pred, y_proba) -> float``.

    Raises:
        UndefinedMetric: If *name* is not a supported metric.
    """
    fn = _METRIC_REGISTRY.get(name)
    if fn is None:
        raise UndefinedMetric(f"Unknown metric {name!r}. Supported: {list(METRIC_NAMES)}")
    return fn


def needs_proba(name: str) -> bool:
    """Return True if *name* requires probability scores.

    Args:
        name: Metric identifier.

    Returns:
        True for ``roc_auc`` and ``pr_auc``; False for label metrics.

    Raises:
        UndefinedMetric: If *name* is not a supported metric.
    """
    if name not in _METRIC_REGISTRY:
        raise UndefinedMetric(f"Unknown metric {name!r}.")
    return name in PROBA_METRICS


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> dict[str, float]:
    """Compute all eight metrics and return them as a dict.

    Args:
        y_true: Ground-truth integer labels, shape (n,), values in {0, 1}.
        y_pred: Predicted integer labels, shape (n,), values in {0, 1}.
        y_proba: Predicted probability for class 1, shape (n,), values in [0, 1].

    Returns:
        Dict mapping metric name to float.  Metrics that cannot be computed
        for the supplied data raise ``UndefinedMetric``; the caller should
        handle those individually if lenient behaviour is desired.

    Raises:
        ValueError: On shape/dimension/value mismatches or invalid inputs.
        UndefinedMetric: If any metric is undefined for the data.
    """
    y_true, y_pred, y_proba = _validate_inputs(y_true, y_pred, y_proba)
    results: dict[str, float] = {}
    for name, fn in _METRIC_REGISTRY.items():
        results[name] = fn(y_true, y_pred, y_proba)
    return results


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def _validate_inputs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Validate and coerce metric inputs.

    Args:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        y_proba: Class-1 probability scores.

    Returns:
        (y_true, y_pred, y_proba) as 1-D numpy arrays.

    Raises:
        ValueError: On invalid inputs.
    """
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()
    y_proba = np.asarray(y_proba, dtype=float).ravel()

    if y_true.ndim != 1 or y_pred.ndim != 1 or y_proba.ndim != 1:
        raise ValueError("y_true, y_pred and y_proba must be 1-D.")

    n = len(y_true)
    if n == 0:
        raise ValueError("Empty arrays are not allowed.")
    if len(y_pred) != n:
        raise ValueError(f"y_pred length {len(y_pred)} != y_true length {n}.")
    if len(y_proba) != n:
        raise ValueError(f"y_proba length {len(y_proba)} != y_true length {n}.")

    # Binary labels
    unique = set(np.unique(y_true).tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(f"y_true must contain only 0 and 1, got: {sorted(unique)}")
    unique_pred = set(np.unique(y_pred).tolist())
    if not unique_pred.issubset({0, 1}):
        raise ValueError(f"y_pred must contain only 0 and 1, got: {sorted(unique_pred)}")

    # Probability scores
    if not np.all(np.isfinite(y_proba)):
        raise ValueError("y_proba contains non-finite values.")
    if np.any(y_proba < 0.0) or np.any(y_proba > 1.0):
        raise ValueError("y_proba values must be in [0, 1].")

    return y_true.astype(int), y_pred.astype(int), y_proba
