"""Dataset profiling: converts validated column information into DatasetProfile (M2).

The profile is computed purely from the raw DataFrame and column_types;
no model fitting or splitting occurs here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelroot.schemas.result import ColumnProfile, DatasetProfile


def profile(
    df: pd.DataFrame,
    column_types: dict[str, str],  # "numeric" | "categorical"
    target_col: str | None = None,
    dropped: list[tuple[str, str]] | None = None,
    warnings: list[str] | None = None,
) -> DatasetProfile:
    """Compute a ``DatasetProfile`` from a raw DataFrame.

    Args:
        df: Raw DataFrame (all columns, including target).
        column_types: Mapping from feature column name to inferred type.
            Columns absent from this mapping are assumed not to be features
            (e.g. the target or dropped columns).
        target_col: Name of the target column; used only for class_balance.
        dropped: List of (column_name, reason) pairs for dropped columns.
        warnings: Pre-existing warnings to include in the profile.

    Returns:
        A ``DatasetProfile`` Pydantic model.
    """
    dropped = dropped or []
    warnings = warnings or []
    n_rows, n_cols_total = df.shape

    # ── Class balance ──────────────────────────────────────────────────────
    class_balance: dict[str, int] = {}
    if target_col is not None and target_col in df.columns:
        counts = df[target_col].replace("", np.nan).dropna().astype(str).value_counts()
        class_balance = {str(k): int(v) for k, v in counts.items()}

    # ── Per-column profiles ────────────────────────────────────────────────
    dropped_names = {name for name, _ in dropped}
    dropped_reasons = {name: reason for name, reason in dropped}

    column_profiles: list[ColumnProfile] = []

    for col in df.columns:
        series = df[col].replace("", np.nan)
        n_total = len(series)
        n_nonnull = int(series.notna().sum())
        missing_frac = 1.0 - n_nonnull / n_total if n_total > 0 else 1.0
        n_unique = int(series.dropna().nunique())
        unique_ratio = n_unique / n_rows if n_rows > 0 else 0.0

        is_dropped = col in dropped_names
        reason = dropped_reasons.get(col)

        # inferred_type: from column_types if feature, else use column's own data
        if col in column_types:
            inferred_type = column_types[col]
        elif is_dropped:
            # Use reason as type hint for dropped columns
            inferred_type = "dropped"
        else:
            # Target or unclassified
            inferred_type = "target" if col == target_col else "unknown"

        column_profiles.append(
            ColumnProfile(
                name=col,
                inferred_type=inferred_type,
                missing_frac=round(missing_frac, 6),
                n_unique=n_unique,
                unique_ratio=round(unique_ratio, 6),
                dropped=is_dropped,
                reason=reason,
            )
        )

    return DatasetProfile(
        n_rows=n_rows,
        n_cols=n_cols_total,
        class_balance=class_balance,
        columns=column_profiles,
        warnings=list(warnings),
    )
