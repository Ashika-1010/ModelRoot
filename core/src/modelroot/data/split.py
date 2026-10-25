"""Train/holdout/current splitting for ModelRoot (M3).

Supports three split modes:
- ``single_file``: Stratified 80/20 split using ``train_test_split``.
- ``split_column``: User supplies a column with values ``"train"`` / ``"test"``.
- ``test_file``: User supplies a separate test CSV file.

Original row indices are retained in all modes for H3b (row-order leakage).
The current dataset (post-deployment data) is optionally aligned to the
feature schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from modelroot.data.errors import ContractViolation, ErrorCode
from modelroot.data.loader import Limits, load_csv
from modelroot.data.validate import ValidatedDataset
from modelroot.schemas.common import SplitMode
from modelroot.schemas.config import AnalysisConfig


@dataclass
class SplitData:
    """Result of splitting a validated dataset into train / holdout / current.

    Attributes:
        X_train: Feature matrix for training (DataFrame of raw strings;
            preprocessing is applied in ``build_preprocessor``).
        y_train: Integer (0/1) target series for training.
        X_holdout: Feature matrix for the holdout set.
        y_holdout: Integer target series for the holdout set.
        X_current: Feature matrix for the optional current dataset.
        y_current: Integer target series for the current dataset (``None``
            if current labels are unavailable).
        orig_index_train: Original DataFrame indices for training rows.
        orig_index_holdout: Original DataFrame indices for holdout rows.
        split_fingerprint: Stable hex string identifying this split.
        warnings: Non-fatal messages generated during splitting.
    """

    X_train: pd.DataFrame
    y_train: pd.Series
    X_holdout: pd.DataFrame
    y_holdout: pd.Series
    X_current: pd.DataFrame | None = None
    y_current: pd.Series | None = None
    orig_index_train: pd.Index = field(default_factory=pd.Index)
    orig_index_holdout: pd.Index = field(default_factory=pd.Index)
    split_fingerprint: str = ""
    warnings: list[str] = field(default_factory=list)


def _split_fingerprint(
    y_train: pd.Series,
    y_holdout: pd.Series,
    orig_index_train: pd.Index,
    orig_index_holdout: pd.Index,
) -> str:
    """Compute a stable fingerprint of the split by hashing row indices."""
    import hashlib

    h = hashlib.sha256()
    h.update(b"train:")
    h.update(np.array(orig_index_train.tolist(), dtype=np.int64).tobytes())
    h.update(b"holdout:")
    h.update(np.array(orig_index_holdout.tolist(), dtype=np.int64).tobytes())
    return h.hexdigest()


def _align_current(
    current_df: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    positive_class: str,
    limits: Limits,
    warnings: list[str],
) -> tuple[pd.DataFrame, pd.Series | None]:
    """Align a current dataset to the feature schema.

    Missing feature columns raise ``ContractViolation(CURRENT_SCHEMA_MISMATCH)``.
    Extra columns are dropped with a warning.  Target is optional.

    Args:
        current_df: Raw current dataset DataFrame.
        feature_cols: Expected feature column names from the training split.
        target_col: Name of the target column.
        positive_class: Positive class string label.
        limits: Resource limits (unused here but passed for consistency).
        warnings: Mutable list to append warnings to.

    Returns:
        Tuple of (X_current, y_current).  y_current is ``None`` if target absent.
    """
    current_cols = set(current_df.columns)
    missing = [c for c in feature_cols if c not in current_cols]
    if missing:
        raise ContractViolation(
            ErrorCode.CURRENT_SCHEMA_MISMATCH,
            f"Current dataset is missing feature columns: {missing}",
        )

    extra = [c for c in current_cols if c not in feature_cols and c != target_col]
    if extra:
        warnings.append(f"Current dataset has extra columns that were ignored: {extra}")

    X_current = current_df[feature_cols].copy()

    y_current: pd.Series | None = None
    if target_col in current_df.columns:
        target_series = current_df[target_col].replace("", float("nan"))
        if target_series.notna().all():
            y_current = (target_series.astype(str) == positive_class).astype(int)
        else:
            warnings.append(
                "Current dataset target column has missing values; treating labels as absent."
            )

    return X_current, y_current


def make_split(
    validated: ValidatedDataset,
    config: AnalysisConfig,
    seed: int,
    limits: Limits | None = None,
) -> SplitData:
    """Split a validated dataset into train / holdout / current sets.

    Args:
        validated: Validated dataset produced by ``validate_for_analysis``.
        config: Analysis configuration (split mode, test_size, etc.).
        seed: Integer seed for the stratified split (from ``SeedManager``).
        limits: Resource limits for loading the current dataset.

    Returns:
        A ``SplitData`` instance with all sets and their original indices.

    Raises:
        ContractViolation: If split constraints are violated.
    """
    if limits is None:
        limits = Limits()

    warnings: list[str] = list(validated.warnings)
    df = validated.df
    feature_cols = validated.feature_cols
    y = validated.y
    split_cfg = config.split

    X = df[feature_cols].copy()

    # â”€â”€ Split modes â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

    if split_cfg.mode == SplitMode.single_file:
        # Stratified 80/20 on row positions; preserve original index
        positions = np.arange(len(df))
        train_pos, holdout_pos = train_test_split(
            positions,
            test_size=split_cfg.test_size,
            stratify=y.to_numpy(),
            random_state=seed,
        )
        train_idx = df.index[train_pos]
        holdout_idx = df.index[holdout_pos]

    elif split_cfg.mode == SplitMode.split_column:
        col = split_cfg.split_column
        assert col is not None
        if col not in df.columns:
            raise ContractViolation(
                ErrorCode.TARGET_MISSING,
                f"Split column '{col}' not found in the dataset.",
            )
        col_vals = df[col].str.strip().str.lower()
        valid_vals = {"train", "test"}
        found_vals = set(col_vals.unique())
        if not found_vals.issubset(valid_vals):
            raise ContractViolation(
                ErrorCode.TARGET_NOT_BINARY,
                f"Split column '{col}' must contain only 'train'/'test'; "
                f"found: {found_vals - valid_vals}",
            )
        if "train" not in found_vals or "test" not in found_vals:
            raise ContractViolation(
                ErrorCode.TARGET_NOT_BINARY,
                f"Split column '{col}' must contain both 'train' and 'test' values.",
            )
        train_idx = df.index[col_vals == "train"]
        holdout_idx = df.index[col_vals == "test"]
        # Remove split column from X
        if col in X.columns:
            X = X.drop(columns=[col])

    elif split_cfg.mode == SplitMode.test_file:
        assert split_cfg.test_path is not None
        test_df_raw = load_csv(split_cfg.test_path, limits=limits)
        # Check schema match
        train_feature_cols = [c for c in feature_cols if c != split_cfg.split_column]
        missing = [c for c in train_feature_cols if c not in test_df_raw.columns]
        if missing:
            raise ContractViolation(
                ErrorCode.CURRENT_SCHEMA_MISMATCH,
                f"Test file is missing columns: {missing}",
            )
        # Training is the full dataset; holdout is the test file
        train_idx = df.index
        # Build holdout from test file rows
        holdout_X_raw = test_df_raw[feature_cols].copy()
        holdout_X_raw.index = pd.RangeIndex(len(df), len(df) + len(test_df_raw))
        holdout_idx = holdout_X_raw.index

        if config.target in test_df_raw.columns:
            holdout_y_raw = (
                test_df_raw[config.target].astype(str) == validated.positive_class
            ).astype(int)
            holdout_y_raw.index = holdout_idx
        else:
            # No target in test file â€” create a dummy (M5 will handle this properly)
            holdout_y_raw = pd.Series(0, index=holdout_idx)

        X_train = X.loc[train_idx]
        y_train = y.loc[train_idx]

        # Handle optional current dataset
        X_current: pd.DataFrame | None = None
        y_current_series: pd.Series | None = None
        if config.current_path is not None:
            current_df_raw = load_csv(config.current_path, limits=limits)
            X_current, y_current_series = _align_current(
                current_df_raw,
                feature_cols,
                config.target,
                validated.positive_class,
                limits,
                warnings,
            )

        fp = _split_fingerprint(y_train, holdout_y_raw, train_idx, holdout_idx)
        return SplitData(
            X_train=X_train,
            y_train=y_train,
            X_holdout=holdout_X_raw,
            y_holdout=holdout_y_raw,
            X_current=X_current,
            y_current=y_current_series,
            orig_index_train=train_idx,
            orig_index_holdout=holdout_idx,
            split_fingerprint=fp,
            warnings=warnings,
        )
    else:
        raise ContractViolation(
            ErrorCode.TARGET_NOT_BINARY,
            f"Unknown split mode: {split_cfg.mode}",
        )

    X_train = X.loc[train_idx]
    y_train = y.loc[train_idx]
    X_holdout = X.loc[holdout_idx]
    y_holdout = y.loc[holdout_idx]

    # â”€â”€ Optional current dataset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    X_current = None
    y_current_series = None
    if config.current_path is not None:
        current_df_raw = load_csv(config.current_path, limits=limits)
        X_current, y_current_series = _align_current(
            current_df_raw,
            feature_cols,
            config.target,
            validated.positive_class,
            limits,
            warnings,
        )

    fp = _split_fingerprint(y_train, y_holdout, train_idx, holdout_idx)

    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        X_current=X_current,
        y_current=y_current_series,
        orig_index_train=train_idx,
        orig_index_holdout=holdout_idx,
        split_fingerprint=fp,
        warnings=warnings,
    )
