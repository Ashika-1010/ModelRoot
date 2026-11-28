"""Subgroup slice definitions, enumeration, and certified exact margin test (M18).

Provides:
- Coordinate-wise monotone raw-difference statistic exact unconditional test.
- Slice representation and deterministic enumeration over categorical and numeric features.
- Mask extraction for train and holdout splits.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats as stats


@dataclass
class Slice:
    """A single candidate subgroup slice.

    Attributes:
        feature: Column name defining the slice.
        value: Category value or string representation of numeric bin.
        is_numeric_bin: True if slice is an interval bin on a continuous numeric feature.
        bin_edges: (low, high) bounds if numeric bin, where low <= X < high.
        n_train: Subgroup observation count on training workspace.
        error_train: Error rate (1 - accuracy) within the slice on training predictions.
        complement_error_train: Error rate on all non-slice training observations.
        gap_train: error_train - complement_error_train.
        p_train: Discovery screening p-value.
        q_train: Discovery screening adjusted q-value.
        n_holdout: Subgroup observation count on holdout set.
        error_holdout: Error rate on holdout set.
        complement_error_holdout: Error rate on holdout complement.
        gap_holdout: error_holdout - complement_error_holdout.
        p_holdout: Holdout confirmation exact margin p-value.
        q_holdout: Holdout confirmation BY-adjusted q-value.
        confirmed: Whether the slice was confirmed under BY FDR control.
        testable: True if n_holdout >= slice_min_n.
    """

    feature: str
    value: Any
    is_numeric_bin: bool = False
    bin_edges: tuple[float, float] | None = None
    n_train: int = 0
    error_train: float = 0.0
    complement_error_train: float = 0.0
    gap_train: float = 0.0
    p_train: float | None = None
    q_train: float | None = None
    n_holdout: int = 0
    error_holdout: float = 0.0
    complement_error_holdout: float = 0.0
    gap_holdout: float = 0.0
    p_holdout: float | None = None
    q_holdout: float | None = None
    confirmed: bool = False
    testable: bool = True

    @property
    def name(self) -> str:
        """Formatted human-readable slice descriptor."""
        if self.is_numeric_bin and self.bin_edges is not None:
            return f"{self.feature} in [{self.bin_edges[0]:.4g}, {self.bin_edges[1]:.4g})"
        return f"{self.feature} == {self.value}"


def exact_margin_test(
    x1: int,
    n1: int,
    x2: int,
    n2: int,
    delta: float = 0.05,
    n_grid: int = 1000,
) -> float:
    """Exact unconditional test for superiority/margin hypothesis using monotone raw difference.

    Tests:
        H0: p1 - p2 <= delta
        H1: p1 - p2 > delta

    Using the coordinate-wise monotone raw difference statistic:
        D(k1, k2) = k1 / n1 - k2 / n2

    Because D is strictly increasing in k1 and decreasing in k2, the rejection region
    R = { (k1, k2): D(k1, k2) >= D_obs } is an exact upper set, mathematically guaranteeing
    that the least-favourable configuration lies on the boundary p1 = p2 + delta.

    To eliminate discretization bias, a certified Lipschitz upper bound is added:
        L = n1 + n2
        p_certified = min(1.0, max_j g(t_j) + (L * (1 - delta)) / (2 * M))

    Args:
        x1: Number of errors in subgroup.
        n1: Total observations in subgroup. Must be >= 1.
        x2: Number of errors in complement.
        n2: Total observations in complement. Must be >= 1.
        delta: Margin threshold (slice_gap > 0).
        n_grid: Number of uniform evaluation intervals M on [0, 1 - delta].

    Returns:
        Certified conservative exact p-value in [0, 1].
    """
    if n1 <= 0 or n2 <= 0:
        raise ValueError(f"Sample sizes must be positive, got n1={n1}, n2={n2}")
    if not (0.0 <= delta < 1.0):
        raise ValueError(f"delta must be in [0, 1), got {delta}")
    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}")

    d_obs = (float(x1) / float(n1)) - (float(x2) / float(n2))

    # Precompute 1D threshold array K1(k2) for each k2 in 0..n2
    # k1 / n1 - k2 / n2 >= d_obs <=> k1 >= n1 * (d_obs + k2 / n2)
    k2 = np.arange(n2 + 1, dtype=np.float64)
    k1_thresholds = np.ceil(float(n1) * (d_obs + k2 / float(n2)) - 1e-12).astype(int)
    k1_min = np.clip(k1_thresholds, 0, n1 + 1)

    # Grid of evaluation points on [0, 1 - delta]
    p2_vals = np.linspace(0.0, 1.0 - delta, n_grid, dtype=np.float64)
    max_g = 0.0

    k2_int = np.arange(n2 + 1)

    for p2 in p2_vals:
        p1 = p2 + delta

        # P(X1 >= k1_min) under Bin(n1, p1)
        if p1 <= 0.0:
            tail1 = (k1_min <= 0).astype(np.float64)
        elif p1 >= 1.0:
            tail1 = (k1_min <= n1).astype(np.float64)
        else:
            tail1 = np.where(
                k1_min <= 0,
                1.0,
                np.where(k1_min > n1, 0.0, stats.binom.sf(k1_min - 1, n1, p1)),
            )

        # PMF of X2 under Bin(n2, p2)
        if p2 <= 0.0:
            pmf2 = np.zeros(n2 + 1, dtype=np.float64)
            pmf2[0] = 1.0
        elif p2 >= 1.0:
            pmf2 = np.zeros(n2 + 1, dtype=np.float64)
            pmf2[n2] = 1.0
        else:
            pmf2 = stats.binom.pmf(k2_int, n2, p2)

        prob = float(np.dot(pmf2, tail1))
        if prob > max_g:
            max_g = prob

    # Conservative Lipschitz upper bound over discretization intervals
    epsilon = ((n1 + n2) * (1.0 - delta)) / (2.0 * (n_grid - 1))
    p_upper = max_g + epsilon

    # Return the conservative upper bound, safely rounded outwards and clipped to [0, 1]
    return float(min(1.0, np.nextafter(p_upper, np.inf)))


def get_slice_mask(df: pd.DataFrame, sl: Slice) -> np.ndarray:
    """Evaluate boolean membership mask for a slice on a dataframe."""
    if sl.feature not in df.columns:
        return np.zeros(len(df), dtype=bool)

    col = df[sl.feature]
    if sl.is_numeric_bin and sl.bin_edges is not None:
        low, high = sl.bin_edges
        # Semi-closed interval: low <= x < high, or x <= high for maximum bin
        mask = col >= low if high == float("inf") else (col >= low) & (col < high)
        return np.asarray(mask, dtype=bool)

    # Categorical level match (handling strings and missing values)
    mask = col == sl.value
    return np.asarray(mask, dtype=bool)


def enumerate_slices(
    df: pd.DataFrame,
    col_types: dict[str, str],
    min_n: int = 30,
    max_features: int = 20,
    max_levels_per_feature: int = 10,
) -> list[Slice]:
    """Enumerate candidate 1D slices from categorical levels and numeric quartile bins.

    Args:
        df: Input training dataframe.
        col_types: Column type mapping ("categorical", "continuous", "numeric", etc.).
        min_n: Minimum observation count for a slice to be testable.
        max_features: Maximum number of features to enumerate.
        max_levels_per_feature: Maximum levels/bins per feature.

    Returns:
        List of candidate Slice objects meeting the size threshold.
    """
    slices: list[Slice] = []
    features_considered = 0

    for col in df.columns:
        if features_considered >= max_features:
            break

        ctype = col_types.get(col, "unknown").lower()
        series = df[col]

        # Categorical features
        if ctype in ("categorical", "string", "category", "bool", "boolean"):
            features_considered += 1
            counts = series.value_counts(dropna=True)
            for val, cnt in counts.head(max_levels_per_feature).items():
                if int(cnt) >= min_n:
                    slices.append(
                        Slice(
                            feature=col,
                            value=val,
                            is_numeric_bin=False,
                            n_train=int(cnt),
                        )
                    )

        # Continuous / numeric features
        elif ctype in ("continuous", "numeric", "float", "integer", "int"):
            clean = series.dropna()
            if len(clean) < min_n * 2:
                continue

            features_considered += 1
            # Quartile binning (4 quantiles)
            try:
                quantiles = np.percentile(clean, [0, 25, 50, 75, 100])
                unique_q = np.unique(quantiles)
                if len(unique_q) < 2:
                    continue

                for i in range(len(unique_q) - 1):
                    low = float(unique_q[i])
                    high = float(unique_q[i + 1])
                    if i == len(unique_q) - 2:
                        # Include right edge for last bin
                        mask = (series >= low) & (series <= high)
                    else:
                        mask = (series >= low) & (series < high)

                    cnt = int(mask.sum())
                    if cnt >= min_n:
                        slices.append(
                            Slice(
                                feature=col,
                                value=f"[{low:.4g}, {high:.4g})",
                                is_numeric_bin=True,
                                bin_edges=(low, high),
                                n_train=cnt,
                            )
                        )
            except Exception:
                continue

    return slices
