"""Multiple testing correction procedures: Benjamini-Hochberg (BH) and Benjamini-Yekutieli (BY).

Provides independent implementations of FDR-controlling procedures under independence/PRDS (BH)
and arbitrary dependence (BY) with rigorous input validation, correct tie handling, and
monotonicity enforcement.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def _validate_p_values(p_values: Sequence[float]) -> np.ndarray:
    """Validate and convert input p-values to a 1D float array."""
    if len(p_values) == 0:
        return np.empty(0, dtype=np.float64)

    arr = np.asarray(p_values, dtype=np.float64)
    if arr.ndim != 1:
        raise ValueError(f"p_values must be a 1D sequence, got shape {arr.shape}")

    if np.any(np.isnan(arr)):
        raise ValueError("p_values must not contain NaN")

    if np.any(np.isinf(arr)):
        raise ValueError("p_values must not contain Inf")

    if np.any((arr < 0.0) | (arr > 1.0)):
        raise ValueError("All p_values must be in the closed interval [0, 1]")

    return arr


def benjamini_hochberg(p_values: Sequence[float]) -> list[float]:
    """Compute Benjamini-Hochberg (BH) FDR-adjusted q-values.

    Controls False Discovery Rate under independence or positive regression dependency (PRDS).

    Args:
        p_values: Sequence of p-values in [0, 1].

    Returns:
        List of adjusted q-values in original input order, clipped to [0, 1].

    Raises:
        ValueError: If any p-value is outside [0, 1], NaN, Inf, or not 1D.
    """
    arr = _validate_p_values(p_values)
    m = len(arr)
    if m == 0:
        return []

    # Sort p-values in ascending order
    sort_indices = np.argsort(arr)
    sorted_p = arr[sort_indices]

    # Ranks 1..m
    ranks = np.arange(1, m + 1, dtype=np.float64)

    # Raw BH adjusted values: p * (m / rank)
    q_raw = sorted_p * (float(m) / ranks)

    # Enforce monotonicity from right to left (descending order): q_{(i)} = min(q_{(i+1)}, q_{(i)})
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]

    # Clip to [0, 1]
    q_clipped = np.clip(q_monotone, 0.0, 1.0)

    # Restore original input order
    result = np.empty(m, dtype=np.float64)
    result[sort_indices] = q_clipped

    return result.tolist()


def benjamini_yekutieli(p_values: Sequence[float]) -> list[float]:
    """Compute Benjamini-Yekutieli (BY) FDR-adjusted q-values.

    Controls False Discovery Rate under arbitrary dependence by incorporating
    the harmonic sum constant c(m) = sum_{k=1}^m (1 / k).

    Args:
        p_values: Sequence of p-values in [0, 1].

    Returns:
        List of adjusted q-values in original input order, clipped to [0, 1].

    Raises:
        ValueError: If any p-value is outside [0, 1], NaN, Inf, or not 1D.
    """
    arr = _validate_p_values(p_values)
    m = len(arr)
    if m == 0:
        return []

    # Harmonic constant c(m) = sum_{k=1}^m 1/k
    c_m = float(np.sum(1.0 / np.arange(1, m + 1, dtype=np.float64)))

    # Sort p-values in ascending order
    sort_indices = np.argsort(arr)
    sorted_p = arr[sort_indices]

    # Ranks 1..m
    ranks = np.arange(1, m + 1, dtype=np.float64)

    # Raw BY adjusted values: p * (m * c_m / rank)
    q_raw = sorted_p * (float(m) * c_m / ranks)

    # Enforce monotonicity from right to left (descending order)
    q_monotone = np.minimum.accumulate(q_raw[::-1])[::-1]

    # Clip to [0, 1]
    q_clipped = np.clip(q_monotone, 0.0, 1.0)

    # Restore original input order
    result = np.empty(m, dtype=np.float64)
    result[sort_indices] = q_clipped

    return result.tolist()


def adjust_p_values(p_values: Sequence[float], method: str = "by") -> list[float]:
    """Adjust p-values using the specified multiple testing procedure.

    Args:
        p_values: Sequence of p-values in [0, 1].
        method: Adjustment method: "bh" or "by" (case-insensitive).

    Returns:
        List of adjusted q-values in original input order.
    """
    norm_method = method.strip().lower()
    if norm_method == "by":
        return benjamini_yekutieli(p_values)
    if norm_method == "bh":
        return benjamini_hochberg(p_values)
    raise ValueError(f"Unknown multiple testing method '{method}'. Supported methods: 'bh', 'by'")
