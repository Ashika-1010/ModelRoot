"""Dataset validation and type inference (M2).

Takes a raw DataFrame from ``load_csv`` and an ``AnalysisConfig`` and
produces a ``ValidatedDataset`` ready for splitting in M3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from modelroot.data.errors import ContractViolation, ErrorCode
from modelroot.schemas.config import AnalysisConfig

#: Minimum number of rows required after target validation.
_MIN_FEATURE_ROWS = 10
#: Minimum number of samples per class.
_MIN_CLASS_SIZE = 2
#: Threshold above which a fraction of numeric-parseable values qualifies as numeric.
_NUMERIC_FRAC = 0.95
#: Minimum distinct non-null values for a column to be considered numeric.
_NUMERIC_UNIQUE_MIN = 10
#: Fraction of non-null values that parse as datetime for a column to be flagged.
_DATETIME_FRAC = 0.95
#: Fraction of missing values above which a high-missingness warning is raised.
_HIGH_MISSING_FRAC = 0.60

ColumnType = Literal["numeric", "categorical"]


@dataclass
class ValidatedDataset:
    """A validated, split-ready dataset.

    Attributes:
        df: The original DataFrame with the target column retained.  Missing
            values in feature columns are preserved; imputation happens in M3.
        feature_cols: Ordered list of feature column names (target excluded,
            dropped columns excluded, id_columns excluded).
        y: Integer target series (0 / 1) with the same index as ``df``.
        positive_class: The original string value that maps to 1.
        column_types: Mapping from feature column name to ``"numeric"`` or
            ``"categorical"``.
        dropped: Names of columns excluded from features (with reasons).
        warnings: Non-fatal informational messages.
    """

    df: pd.DataFrame
    feature_cols: list[str]
    y: pd.Series
    positive_class: str
    column_types: dict[str, ColumnType]
    dropped: list[tuple[str, str]]  # (name, reason)
    warnings: list[str]


def _try_numeric(series: pd.Series) -> float:
    """Return fraction of non-null values in *series* that parse as float."""
    non_null = series.replace("", np.nan).dropna()
    if len(non_null) == 0:
        return 0.0
    converted = pd.to_numeric(non_null, errors="coerce")
    return float(converted.notna().sum()) / len(non_null)


def _try_datetime(series: pd.Series) -> float:
    """Return fraction of non-null values that parse as datetime."""
    non_null = series.replace("", np.nan).dropna()
    if len(non_null) == 0:
        return 0.0
    try:
        converted = pd.to_datetime(non_null, errors="coerce")
        return float(converted.notna().sum()) / len(non_null)
    except Exception:
        return 0.0


def _infer_column_types(
    df: pd.DataFrame,
    exclude_cols: set[str],
    warnings: list[str],
    dropped: list[tuple[str, str]],
) -> tuple[dict[str, ColumnType], list[str]]:
    """Infer column types and identify columns to drop.

    Args:
        df: Raw string DataFrame (target column still present).
        exclude_cols: Column names to skip (e.g. target, already-dropped).
        warnings: Mutable list to append non-fatal warnings to.
        dropped: Mutable list to append (name, reason) pairs to.

    Returns:
        Tuple of (column_types dict, feature_cols list).
    """
    column_types: dict[str, ColumnType] = {}
    feature_cols: list[str] = []

    for col in df.columns:
        if col in exclude_cols:
            continue

        series = df[col]
        non_null = series.replace("", np.nan).dropna()
        n_total = len(series)
        n_nonnull = len(non_null)
        missing_frac = 1.0 - n_nonnull / n_total if n_total > 0 else 1.0

        # Constant column â€” drop
        if n_nonnull > 0 and non_null.nunique() == 1:
            dropped.append((col, "constant column"))
            warnings.append(f"Column '{col}' is constant and was dropped.")
            continue

        # Datetime â€” drop with warning
        if _try_datetime(series) >= _DATETIME_FRAC:
            dropped.append((col, "datetime-like column"))
            warnings.append(
                f"Column '{col}' appears to be datetime-like and was excluded from features."
            )
            continue

        # High missingness â€” warn but keep
        if missing_frac > _HIGH_MISSING_FRAC:
            warnings.append(f"Column '{col}' has {missing_frac:.0%} missing values.")

        # Type inference
        numeric_frac = _try_numeric(series)
        n_unique = int(non_null.nunique()) if n_nonnull > 0 else 0

        if numeric_frac >= _NUMERIC_FRAC and n_unique > _NUMERIC_UNIQUE_MIN:
            column_types[col] = "numeric"
        else:
            column_types[col] = "categorical"

        feature_cols.append(col)

    return column_types, feature_cols


def validate_for_analysis(
    df: pd.DataFrame,
    config: AnalysisConfig,
) -> ValidatedDataset:
    """Validate a raw DataFrame against an ``AnalysisConfig``.

    Performs target extraction, class validation, column type inference,
    constant/datetime column dropping, ID-column exclusion, and
    high-missingness warnings.

    Args:
        df: Raw string DataFrame produced by ``load_csv``.
        config: Analysis configuration specifying target, positive class,
            and metadata hints.

    Returns:
        A ``ValidatedDataset`` ready for M3 splitting.

    Raises:
        ContractViolation: If any validation rule is violated.
    """
    warnings: list[str] = []
    dropped: list[tuple[str, str]] = []

    # â”€â”€ Target column â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if config.target not in df.columns:
        raise ContractViolation(
            ErrorCode.TARGET_MISSING,
            f"Target column '{config.target}' not found in the dataset.",
        )

    target_series = df[config.target].replace("", np.nan)

    # â”€â”€ Target nulls â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if target_series.isna().any():
        raise ContractViolation(
            ErrorCode.TARGET_HAS_NULLS,
            f"Target column '{config.target}' contains missing values.",
        )

    # â”€â”€ Binary target â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    unique_classes = set(target_series.dropna().astype(str).unique())
    if len(unique_classes) != 2:
        raise ContractViolation(
            ErrorCode.TARGET_NOT_BINARY,
            f"Target column must have exactly 2 unique values; "
            f"found {len(unique_classes)}: {sorted(unique_classes)}",
        )

    # â”€â”€ Positive class â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    pos = str(config.positive_class)
    if pos not in unique_classes:
        raise ContractViolation(
            ErrorCode.POSITIVE_CLASS_INVALID,
            f"positive_class '{pos}' not found among target values {sorted(unique_classes)}.",
        )

    # â”€â”€ Encode y as 0/1 â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    y = (target_series.astype(str) == pos).astype(int)
    y.name = config.target

    # â”€â”€ Minimum row count â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if len(df) < _MIN_FEATURE_ROWS:
        raise ContractViolation(
            ErrorCode.TOO_FEW_ROWS,
            f"Dataset has only {len(df)} rows; minimum is {_MIN_FEATURE_ROWS}.",
        )

    # â”€â”€ Class size â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    class_counts = y.value_counts()
    for cls_val, cnt in class_counts.items():
        if cnt < _MIN_CLASS_SIZE:
            label = pos if cls_val == 1 else "negative"
            raise ContractViolation(
                ErrorCode.CLASS_TOO_SMALL,
                f"Class '{label}' has only {cnt} sample(s); minimum is {_MIN_CLASS_SIZE}.",
            )

    # â”€â”€ ID columns â€” exclude with warning â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    id_cols = set(config.metadata.id_columns)
    for id_col in config.metadata.id_columns:
        if id_col in df.columns and id_col != config.target:
            dropped.append((id_col, "declared id_column"))
            warnings.append(f"Column '{id_col}' is declared as an id_column and was excluded.")

    # â”€â”€ Column type inference (skip target and id_columns) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    exclude = {config.target} | id_cols
    column_types, feature_cols = _infer_column_types(df, exclude, warnings, dropped)

    return ValidatedDataset(
        df=df,
        feature_cols=feature_cols,
        y=y,
        positive_class=pos,
        column_types=column_types,
        dropped=dropped,
        warnings=warnings,
    )
