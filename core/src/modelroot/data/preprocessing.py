"""Preprocessing pipeline builder (M3).

Builds an *unfitted* scikit-learn ``ColumnTransformer`` that encodes the
full feature set.  The transformer is fitted on training data only (inside
M4 ``ModelAdapter.fit``).

Numeric features:
  - String-to-float conversion (handles empty strings as NaN)
  - Median imputation (``SimpleImputer(strategy="median")``)
  - Optional standard scaling (``StandardScaler``)

Categorical features:
  - Most-frequent imputation (``SimpleImputer(strategy="most_frequent")``)
  - One-hot encoding with unknown-category handling and frequency/cardinality
    limits.

The transformer is compatible with all four planned sklearn estimators:
LogisticRegression, DecisionTree, RandomForest, HistGradientBoosting.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ColumnType = Literal["numeric", "categorical"]


class _StringToFloat(BaseEstimator, TransformerMixin):  # type: ignore[misc]
    """Convert a string DataFrame/array column to float, mapping empty strings to NaN.

    This is needed because ``load_csv`` keeps all values as strings;
    scikit-learn's ``SimpleImputer`` with ``strategy="median"`` requires
    numeric input.
    """

    def fit(self, X: object, y: object = None) -> _StringToFloat:
        """No-op fit."""
        self._fitted = True
        return self

    def transform(self, X: object, y: object = None) -> np.ndarray:
        """Convert *X* to float, replacing empty strings and non-numeric values with NaN.

        Args:
            X: Input array or DataFrame of strings.

        Returns:
            2D float NumPy array with NaN for missing/non-numeric values.
        """
        import pandas as _pd

        if isinstance(X, _pd.DataFrame):
            result = np.asarray(
                X.replace("", np.nan).apply(_pd.to_numeric, errors="coerce").to_numpy(),
                dtype=float,
            )
        else:
            arr = np.asarray(X, dtype=object)
            result = np.empty(arr.shape, dtype=float)
            for idx in np.ndindex(arr.shape):
                v = arr[idx]
                if v == "" or v is None:
                    result[idx] = np.nan
                else:
                    try:
                        result[idx] = float(v)
                    except (ValueError, TypeError):
                        result[idx] = np.nan
        return result


def build_preprocessor(
    column_types: dict[str, str],
    scale: bool = False,
) -> ColumnTransformer:
    """Build an unfitted ``ColumnTransformer`` for the given feature columns.

    Args:
        column_types: Mapping from feature column name to ``"numeric"`` or
            ``"categorical"``.  The transformer will be built for exactly
            these columns in the order they appear in the mapping.
        scale: If ``True``, apply ``StandardScaler`` to numeric features
            after imputation.  Set to ``True`` for LogisticRegression;
            ``False`` for tree-based models.

    Returns:
        An unfitted ``ColumnTransformer``.  Call ``.fit_transform(X_train)``
        or ``.fit(X_train).transform(X_holdout)`` on it.

    Raises:
        ValueError: If ``column_types`` is empty.
    """
    if not column_types:
        raise ValueError("column_types must not be empty")

    numeric_cols = [c for c, t in column_types.items() if t == "numeric"]
    categorical_cols = [c for c, t in column_types.items() if t == "categorical"]

    transformers: list[tuple[str, object, list[str]]] = []

    # ── Numeric pipeline ──────────────────────────────────────────────────
    if numeric_cols:
        num_steps: list[tuple[str, object]] = [
            ("to_float", _StringToFloat()),
            ("imputer", SimpleImputer(strategy="median")),
        ]
        if scale:
            num_steps.append(("scaler", StandardScaler()))
        numeric_pipeline = Pipeline(steps=num_steps)
        transformers.append(("numeric", numeric_pipeline, numeric_cols))

    # ── Categorical pipeline ──────────────────────────────────────────────
    if categorical_cols:
        cat_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OneHotEncoder(
                        handle_unknown="ignore",
                        min_frequency=0.01,
                        max_categories=50,
                        sparse_output=False,
                    ),
                ),
            ]
        )
        transformers.append(("categorical", cat_pipeline, categorical_cols))

    ct = ColumnTransformer(
        transformers=transformers,
        remainder="drop",  # drop any columns not in column_types
        verbose_feature_names_out=False,
    )
    return ct
