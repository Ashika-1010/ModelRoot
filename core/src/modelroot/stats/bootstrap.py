"""Paired and two-sample bootstrap confidence intervals (M7).

Key contracts
-------------
- ``delta = metric_b - metric_a`` (always).
- Resampling block size = 100; resamples are drawn in blocks of 100 indices
  to avoid extreme per-iteration overhead.
- Shared resample indices for paired bootstrap.
- Independent resampling for two-sample bootstrap.
- Percentile CI via ``numpy.quantile(..., method="linear")``.
- Sample std (ddof=1).
- ``BootstrapError`` raised only when *strictly more than* 20 % of resamples
  are invalid.
- Vectorised fast path for the six label metrics.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelroot.assessment.metrics import (
    LABEL_METRICS,
    UndefinedMetric,
    metric_fn,
    needs_proba,
)

_BLOCK = 100  # resampling block size


# ---------------------------------------------------------------------------
# Public types
# ---------------------------------------------------------------------------


class BootstrapError(Exception):
    """Raised when too many bootstrap resamples are invalid.

    Specifically raised when *strictly more than* 20 % of resamples yield
    an undefined metric (e.g. single class in AUC computation).
    """


@dataclass(frozen=True)
class BootstrapResult:
    """Result of a paired or two-sample bootstrap.

    Attributes:
        delta: Point estimate ``metric_b - metric_a``.
        ci_low: Lower bound of the bootstrap CI.
        ci_high: Upper bound of the bootstrap CI.
        se: Bootstrap standard error (sample std of resample deltas, ddof=1).
        n_valid: Number of valid resamples used.
        n_boot: Total resamples attempted.
        metric_a: Whole-sample metric for the first set / model.
        metric_b: Whole-sample metric for the second set / model.
    """

    delta: float
    ci_low: float
    ci_high: float
    se: float
    n_valid: int
    n_boot: int
    metric_a: float
    metric_b: float


@dataclass(frozen=True)
class CIResult:
    """Result of a single-sample bootstrap CI.

    Attributes:
        estimate: Whole-sample metric value.
        ci_low: Lower bound of the bootstrap CI.
        ci_high: Upper bound of the bootstrap CI.
        se: Bootstrap standard error (sample std, ddof=1).
        n_valid: Number of valid resamples.
        n_boot: Total resamples attempted.
    """

    estimate: float
    ci_low: float
    ci_high: float
    se: float
    n_valid: int
    n_boot: int


# ---------------------------------------------------------------------------
# Vectorised fast path for label metrics
# ---------------------------------------------------------------------------


def _fast_label_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    metric: str,
    idx: np.ndarray,
) -> float | None:
    """Compute metric_b - metric_a on resample *idx* using fast numpy ops.

    Returns ``None`` if the resample yields only one class in y_true.

    Args:
        y_true: Full ground-truth array.
        y_pred_a: Full predictions for model a.
        y_pred_b: Full predictions for model b.
        metric: Metric name (must be in ``LABEL_METRICS``).
        idx: 1-D array of resample row indices.

    Returns:
        Float delta or None if undefined.
    """
    yt = y_true[idx]
    if len(np.unique(yt)) < 2 and metric != "accuracy":
        return None
    ya = y_pred_a[idx]
    yb = y_pred_b[idx]

    if metric == "accuracy":
        return float(np.mean(yb == yt)) - float(np.mean(ya == yt))

    # For binary label metrics, compute TP/FP/FN directly
    def _binary_stats(y: np.ndarray, t: np.ndarray) -> tuple[int, int, int]:
        tp = int(np.sum((y == 1) & (t == 1)))
        fp = int(np.sum((y == 1) & (t == 0)))
        fn = int(np.sum((y == 0) & (t == 1)))
        return tp, fp, fn

    if metric == "balanced_accuracy":
        # balanced_accuracy = (recall_pos + recall_neg) / 2
        def _ba(y: np.ndarray, t: np.ndarray) -> float:
            tp = float(np.sum((y == 1) & (t == 1)))
            tn = float(np.sum((y == 0) & (t == 0)))
            pos = float(np.sum(t == 1))
            neg = float(np.sum(t == 0))
            recall_p = tp / pos if pos > 0 else 0.0
            recall_n = tn / neg if neg > 0 else 0.0
            return (recall_p + recall_n) / 2.0

        return _ba(yb, yt) - _ba(ya, yt)

    if metric in ("macro_f1",):
        # macro_f1 = average of F1 for each class
        def _macro_f1(y: np.ndarray, t: np.ndarray) -> float:
            f1s = []
            for cls in [0, 1]:
                tp = float(np.sum((y == cls) & (t == cls)))
                fp = float(np.sum((y == cls) & (t != cls)))
                fn = float(np.sum((y != cls) & (t == cls)))
                denom = 2 * tp + fp + fn
                f1s.append(2 * tp / denom if denom > 0 else 0.0)
            return float(np.mean(f1s))

        return _macro_f1(yb, yt) - _macro_f1(ya, yt)

    if metric == "f1":
        tp_a, fp_a, fn_a = _binary_stats(ya, yt)
        tp_b, fp_b, fn_b = _binary_stats(yb, yt)
        f1_a = 2 * tp_a / (2 * tp_a + fp_a + fn_a) if (2 * tp_a + fp_a + fn_a) > 0 else 0.0
        f1_b = 2 * tp_b / (2 * tp_b + fp_b + fn_b) if (2 * tp_b + fp_b + fn_b) > 0 else 0.0
        return f1_b - f1_a

    if metric == "precision":
        tp_a, fp_a, _ = _binary_stats(ya, yt)
        tp_b, fp_b, _ = _binary_stats(yb, yt)
        p_a = tp_a / (tp_a + fp_a) if (tp_a + fp_a) > 0 else 0.0
        p_b = tp_b / (tp_b + fp_b) if (tp_b + fp_b) > 0 else 0.0
        return p_b - p_a

    if metric == "recall":
        tp_a, _, fn_a = _binary_stats(ya, yt)
        tp_b, _, fn_b = _binary_stats(yb, yt)
        r_a = tp_a / (tp_a + fn_a) if (tp_a + fn_a) > 0 else 0.0
        r_b = tp_b / (tp_b + fn_b) if (tp_b + fn_b) > 0 else 0.0
        return r_b - r_a

    return None  # Fallback (should never reach here for LABEL_METRICS)


def _generic_delta(
    y_true: np.ndarray,
    y_pred_a: np.ndarray,
    y_pred_b: np.ndarray,
    y_proba_a: np.ndarray,
    y_proba_b: np.ndarray,
    metric: str,
    idx: np.ndarray,
) -> float | None:
    """Compute metric_b - metric_a via the M5 metric registry.

    Returns ``None`` if the resample is invalid.

    Args:
        y_true: Full ground-truth array.
        y_pred_a: Predictions for model a.
        y_pred_b: Predictions for model b.
        y_proba_a: Probabilities for model a.
        y_proba_b: Probabilities for model b.
        metric: Metric name.
        idx: Resample indices.

    Returns:
        Float delta or None.
    """
    yt = y_true[idx]
    ya = y_pred_a[idx]
    yb = y_pred_b[idx]
    pa = y_proba_a[idx]
    pb = y_proba_b[idx]
    fn = metric_fn(metric)
    try:
        va = fn(yt, ya, pa)
        vb = fn(yt, yb, pb)
        return vb - va
    except UndefinedMetric:
        return None


# ---------------------------------------------------------------------------
# paired_bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(
    y_true: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    score_a: float | None = None,
    score_b: float | None = None,
    y_proba_a: np.ndarray | None = None,
    y_proba_b: np.ndarray | None = None,
) -> BootstrapResult:
    """Paired bootstrap CI for the difference ``metric_b - metric_a``.

    The same resample indices are applied to both models.

    Args:
        y_true: Ground-truth labels (0/1), shape (n,).
        pred_a: Predictions for model a (0/1), shape (n,).
        pred_b: Predictions for model b (0/1), shape (n,).
        metric: Metric name (must be in ``METRIC_NAMES``).
        n_boot: Number of bootstrap resamples (>= 100).
        seed: RNG seed.
        alpha: CI significance level (0 < alpha < 1).
        score_a: Pre-computed whole-sample metric for a; computed if None.
        score_b: Pre-computed whole-sample metric for b; computed if None.
        y_proba_a: Class-1 probabilities for model a (required for AUC metrics).
        y_proba_b: Class-1 probabilities for model b (required for AUC metrics).

    Returns:
        :class:`BootstrapResult`.

    Raises:
        ValueError: On invalid inputs.
        BootstrapError: If > 20 % of resamples are invalid.
        UndefinedMetric: If the metric is unknown.
    """
    _check_metric(metric)
    _check_n_boot(n_boot)
    _check_alpha(alpha)

    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    pred_a_arr = np.asarray(pred_a, dtype=int).ravel()
    pred_b_arr = np.asarray(pred_b, dtype=int).ravel()
    _check_lengths(y_true_arr, pred_a_arr, pred_b_arr)

    use_proba = needs_proba(metric)
    y_proba_a_arr = _coerce_proba(y_proba_a, len(y_true_arr), "y_proba_a", required=use_proba)
    y_proba_b_arr = _coerce_proba(y_proba_b, len(y_true_arr), "y_proba_b", required=use_proba)

    # Whole-sample scores
    if score_a is None or score_b is None:
        fn = metric_fn(metric)
        if score_a is None:
            score_a = fn(y_true_arr, pred_a_arr, y_proba_a_arr)
        if score_b is None:
            score_b = fn(y_true_arr, pred_b_arr, y_proba_b_arr)
    whole_delta = score_b - score_a

    rng = np.random.default_rng(seed)
    n = len(y_true_arr)
    deltas: list[float] = []
    n_invalid = 0

    is_label = metric in LABEL_METRICS

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        if is_label:
            d = _fast_label_delta(y_true_arr, pred_a_arr, pred_b_arr, metric, idx)
        else:
            d = _generic_delta(
                y_true_arr, pred_a_arr, pred_b_arr, y_proba_a_arr, y_proba_b_arr, metric, idx
            )
        if d is None:
            n_invalid += 1
        else:
            deltas.append(d)

    if n_invalid > 0.2 * n_boot:
        raise BootstrapError(
            f"{n_invalid}/{n_boot} resamples were invalid (> 20 %); "
            "data may be severely imbalanced or too small."
        )

    if not deltas:
        raise BootstrapError("All bootstrap resamples were invalid.")

    arr = np.array(deltas, dtype=float)
    ci_low = float(np.quantile(arr, alpha / 2, method="linear"))
    ci_high = float(np.quantile(arr, 1.0 - alpha / 2, method="linear"))
    se = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    return BootstrapResult(
        delta=whole_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        n_valid=len(deltas),
        n_boot=n_boot,
        metric_a=float(score_a),
        metric_b=float(score_b),
    )


# ---------------------------------------------------------------------------
# bootstrap_ci
# ---------------------------------------------------------------------------


def bootstrap_ci(
    y_true: np.ndarray,
    pred: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    y_proba: np.ndarray | None = None,
) -> CIResult:
    """Single-sample bootstrap CI for a metric.

    Args:
        y_true: Ground-truth labels (0/1), shape (n,).
        pred: Predicted labels (0/1), shape (n,).
        metric: Metric name.
        n_boot: Number of bootstrap resamples (>= 100).
        seed: RNG seed.
        alpha: CI significance level.
        y_proba: Class-1 probabilities (required for AUC metrics).

    Returns:
        :class:`CIResult`.

    Raises:
        ValueError: On invalid inputs.
        BootstrapError: If > 20 % of resamples are invalid.
    """
    _check_metric(metric)
    _check_n_boot(n_boot)
    _check_alpha(alpha)

    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    pred_arr = np.asarray(pred, dtype=int).ravel()
    if len(y_true_arr) != len(pred_arr):
        raise ValueError("y_true and pred must have the same length.")

    use_proba = needs_proba(metric)
    y_proba_arr = _coerce_proba(y_proba, len(y_true_arr), "y_proba", required=use_proba)

    fn = metric_fn(metric)
    estimate = fn(y_true_arr, pred_arr, y_proba_arr)

    rng = np.random.default_rng(seed)
    n = len(y_true_arr)
    values: list[float] = []
    n_invalid = 0

    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yt = y_true_arr[idx]
        yp = pred_arr[idx]
        ypr = y_proba_arr[idx]
        try:
            values.append(fn(yt, yp, ypr))
        except UndefinedMetric:
            n_invalid += 1

    if n_invalid > 0.2 * n_boot:
        raise BootstrapError(f"{n_invalid}/{n_boot} resamples were invalid (> 20 %).")

    if not values:
        raise BootstrapError("All bootstrap resamples were invalid.")

    arr = np.array(values, dtype=float)
    ci_low = float(np.quantile(arr, alpha / 2, method="linear"))
    ci_high = float(np.quantile(arr, 1.0 - alpha / 2, method="linear"))
    se = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    return CIResult(
        estimate=float(estimate),
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        n_valid=len(values),
        n_boot=n_boot,
    )


# ---------------------------------------------------------------------------
# two_sample_bootstrap_diff
# ---------------------------------------------------------------------------


def two_sample_bootstrap_diff(
    y_true_a: np.ndarray,
    pred_a: np.ndarray,
    y_true_b: np.ndarray,
    pred_b: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    y_proba_a: np.ndarray | None = None,
    y_proba_b: np.ndarray | None = None,
) -> BootstrapResult:
    """Two-sample bootstrap CI for ``metric_b - metric_a``.

    Each dataset is resampled independently.  Datasets may have different
    sizes.

    Args:
        y_true_a: Ground-truth for set a.
        pred_a: Predictions for set a.
        y_true_b: Ground-truth for set b.
        pred_b: Predictions for set b.
        metric: Metric name.
        n_boot: Number of bootstrap resamples (>= 100).
        seed: RNG seed.
        alpha: CI significance level.
        y_proba_a: Class-1 probabilities for set a.
        y_proba_b: Class-1 probabilities for set b.

    Returns:
        :class:`BootstrapResult`.

    Raises:
        ValueError: On invalid inputs.
        BootstrapError: If > 20 % of resamples are invalid.
    """
    _check_metric(metric)
    _check_n_boot(n_boot)
    _check_alpha(alpha)

    ya = np.asarray(y_true_a, dtype=int).ravel()
    pa = np.asarray(pred_a, dtype=int).ravel()
    yb = np.asarray(y_true_b, dtype=int).ravel()
    pb = np.asarray(pred_b, dtype=int).ravel()

    if len(ya) != len(pa):
        raise ValueError("y_true_a and pred_a must have the same length.")
    if len(yb) != len(pb):
        raise ValueError("y_true_b and pred_b must have the same length.")
    if len(ya) == 0 or len(yb) == 0:
        raise ValueError("Datasets must not be empty.")

    use_proba = needs_proba(metric)
    pa_arr = _coerce_proba(y_proba_a, len(ya), "y_proba_a", required=use_proba)
    pb_arr = _coerce_proba(y_proba_b, len(yb), "y_proba_b", required=use_proba)

    fn = metric_fn(metric)
    score_a = fn(ya, pa, pa_arr)
    score_b = fn(yb, pb, pb_arr)
    whole_delta = score_b - score_a

    rng = np.random.default_rng(seed)
    na, nb = len(ya), len(yb)
    deltas: list[float] = []
    n_invalid = 0

    for _ in range(n_boot):
        idx_a = rng.integers(0, na, size=na)
        idx_b = rng.integers(0, nb, size=nb)
        try:
            va = fn(ya[idx_a], pa[idx_a], pa_arr[idx_a])
            vb = fn(yb[idx_b], pb[idx_b], pb_arr[idx_b])
            deltas.append(vb - va)
        except UndefinedMetric:
            n_invalid += 1

    if n_invalid > 0.2 * n_boot:
        raise BootstrapError(f"{n_invalid}/{n_boot} resamples were invalid (> 20 %).")

    if not deltas:
        raise BootstrapError("All bootstrap resamples were invalid.")

    arr = np.array(deltas, dtype=float)
    ci_low = float(np.quantile(arr, alpha / 2, method="linear"))
    ci_high = float(np.quantile(arr, 1.0 - alpha / 2, method="linear"))
    se = float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0

    return BootstrapResult(
        delta=whole_delta,
        ci_low=ci_low,
        ci_high=ci_high,
        se=se,
        n_valid=len(deltas),
        n_boot=n_boot,
        metric_a=float(score_a),
        metric_b=float(score_b),
    )


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _check_metric(name: str) -> None:
    if not needs_proba.__module__ or name not in (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
    ):
        # Validate via metric_fn which raises UndefinedMetric on unknown names
        metric_fn(name)


def _check_n_boot(n_boot: int) -> None:
    if n_boot < 100:
        raise ValueError(f"n_boot must be >= 100, got {n_boot}.")


def _check_alpha(alpha: float) -> None:
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")


def _check_lengths(*arrays: np.ndarray) -> None:
    lengths = [len(a) for a in arrays]
    if len(set(lengths)) > 1:
        raise ValueError(f"All arrays must have the same length; got {lengths}.")
    if lengths[0] == 0:
        raise ValueError("Arrays must not be empty.")


def _coerce_proba(
    y_proba: np.ndarray | None,
    n: int,
    name: str,
    required: bool,
) -> np.ndarray:
    """Return a valid probability array.

    If *y_proba* is None and not *required*, returns zeros.

    Args:
        y_proba: Input array or None.
        n: Expected length.
        name: Parameter name for error messages.
        required: Whether probabilities are required for this metric.

    Returns:
        1-D float array of length n.

    Raises:
        ValueError: If required but None, or if shape/values are invalid.
    """
    if y_proba is None:
        if required:
            raise ValueError(f"{name} is required for probability-based metrics.")
        return np.zeros(n, dtype=float)
    arr = np.asarray(y_proba, dtype=float).ravel()
    if len(arr) != n:
        raise ValueError(f"{name} length {len(arr)} != y_true length {n}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains non-finite values.")
    if np.any(arr < 0.0) or np.any(arr > 1.0):
        raise ValueError(f"{name} values must be in [0, 1].")
    return arr
