"""Paired subset bootstrap confidence intervals (M15).

Compares model performance on the full dataset vs a designated subset
(e.g., non-duplicate holdout rows).

Key contracts:
- Resample the same holdout indices for both metrics.
- delta = metric(all rows) - metric(subset rows).
- Percentile CI, ddof=1 standard error.
- Resampling block size = 100.
- Preserves BootstrapResult conventions: metric_a=subset, metric_b=full.
"""

from __future__ import annotations

import numpy as np

from modelroot.assessment.metrics import (
    UndefinedMetric,
    metric_fn,
    needs_proba,
)
from modelroot.stats.bootstrap import (
    BootstrapError,
    BootstrapResult,
    _check_alpha,
    _check_metric,
    _check_n_boot,
    _coerce_proba,
)

_BLOCK = 100


def paired_subset_bootstrap(
    y_true: np.ndarray,
    pred: np.ndarray,
    mask: np.ndarray,
    metric: str,
    n_boot: int = 1000,
    seed: int = 0,
    alpha: float = 0.05,
    y_proba: np.ndarray | None = None,
) -> BootstrapResult:
    """Paired subset bootstrap CI for delta = metric(all rows) - metric(subset rows).

    Resamples the same holdout indices for both metrics.

    Args:
        y_true: Ground-truth binary labels (0/1), shape (n,).
        pred: Predicted binary labels (0/1), shape (n,).
        mask: Boolean mask of shape (n,) where True indicates subset rows.
        metric: Metric name.
        n_boot: Number of bootstrap resamples (>= 100).
        seed: RNG seed.
        alpha: CI significance level.
        y_proba: Class-1 probabilities (required for AUC metrics).

    Returns:
        :class:`BootstrapResult` with metric_a=subset, metric_b=full.

    Raises:
        ValueError: On invalid inputs (length mismatch, non-binary labels, < 2 subset rows).
        BootstrapError: If strictly more than 20% of resamples are invalid.
    """
    _check_metric(metric)
    _check_n_boot(n_boot)
    _check_alpha(alpha)

    y_true_arr = np.asarray(y_true, dtype=int).ravel()
    pred_arr = np.asarray(pred, dtype=int).ravel()
    mask_arr = np.asarray(mask, dtype=bool).ravel()

    n = len(y_true_arr)
    if len(pred_arr) != n or len(mask_arr) != n:
        raise ValueError("y_true, pred, and mask must have the same length.")

    unique_labels = set(np.unique(y_true_arr))
    if not unique_labels.issubset({0, 1}):
        raise ValueError(f"y_true must contain only binary labels 0 and 1, got {unique_labels}")

    n_subset = int(np.sum(mask_arr))
    if n_subset < 2:
        raise ValueError(f"mask must contain at least two True rows, got {n_subset}")

    use_proba = needs_proba(metric)
    y_proba_arr = _coerce_proba(y_proba, n, "y_proba", required=use_proba)

    fn = metric_fn(metric)

    # Whole-sample scores
    # metric_a is subset, metric_b is full
    try:
        score_full = float(fn(y_true_arr, pred_arr, y_proba_arr))
    except UndefinedMetric as err:
        raise ValueError(f"Metric {metric!r} is undefined on full set: {err}") from err

    y_sub = y_true_arr[mask_arr]
    p_sub = pred_arr[mask_arr]
    pr_sub = y_proba_arr[mask_arr]
    try:
        score_subset = float(fn(y_sub, p_sub, pr_sub))
    except UndefinedMetric as err:
        raise ValueError(f"Metric {metric!r} is undefined on subset: {err}") from err

    whole_delta = score_full - score_subset

    rng = np.random.default_rng(seed)
    deltas: list[float] = []
    n_invalid = 0

    # Resample in blocks of 100
    n_blocks = (n_boot + _BLOCK - 1) // _BLOCK
    samples_drawn = 0

    for _ in range(n_blocks):
        current_block = min(_BLOCK, n_boot - samples_drawn)
        samples_drawn += current_block
        for _ in range(current_block):
            idx = rng.integers(0, n, size=n)
            sub_idx = idx[mask_arr[idx]]

            if len(sub_idx) < 2 or len(np.unique(y_true_arr[sub_idx])) < 2:
                n_invalid += 1
                continue

            try:
                f_full = fn(y_true_arr[idx], pred_arr[idx], y_proba_arr[idx])
                f_sub = fn(y_true_arr[sub_idx], pred_arr[sub_idx], y_proba_arr[sub_idx])
                deltas.append(float(f_full - f_sub))
            except UndefinedMetric:
                n_invalid += 1

    if n_invalid > 0.2 * n_boot:
        raise BootstrapError(
            f"{n_invalid}/{n_boot} resamples were invalid (> 20 %); "
            "subset may be too small or lack class diversity."
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
        metric_a=score_subset,
        metric_b=score_full,
    )
