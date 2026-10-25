"""ModelAdapter protocol and supporting types (M4).

All adapters wrap a scikit-learn ``Pipeline`` that combines:
  - The M3 ``ColumnTransformer`` preprocessor (fitted on training data).
  - The estimator.

Public interface used by the experiment runner (M9+):
  - ``fit(X, y, sample_weight=None)``
  - ``predict(X)`` ->' integer array of 0/1 labels
  - ``predict_proba(X)`` ->' (n, 2) float array
  - ``clone_with(**overrides)`` ->' new adapter, original unchanged
  - ``capacity_grid()`` ->' list[dict] of param dicts for grid search
  - ``supports_sample_weight()`` ->' bool
  - ``complexity()`` ->' float (higher = more complex / more prone to overfit)
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np
import pandas as pd


class ModelAdapter(ABC):
    """Abstract base class for all ModelRoot model adapters.

    Each concrete adapter wraps exactly one sklearn estimator and one
    preprocessing ``ColumnTransformer`` in an sklearn ``Pipeline``.
    """

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> ModelAdapter:
        """Fit the model on training data.

        Args:
            X: Training feature DataFrame (raw strings; preprocessing is
               applied inside the pipeline).
            y: Integer (0/1) target Series.
            sample_weight: Optional per-sample weight array.  Ignored if
               ``supports_sample_weight()`` returns ``False``.

        Returns:
            ``self`` for chaining.

        Raises:
            ValueError: If fit parameters are invalid.
        """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return predicted class labels (0 or 1).

        Args:
            X: Feature DataFrame (same schema as training data).

        Returns:
            1-D integer array of shape (n,).
        """

    @abstractmethod
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return class probabilities.

        Args:
            X: Feature DataFrame (same schema as training data).

        Returns:
            2-D float array of shape (n, 2) where column 0 is P(class=0)
            and column 1 is P(class=1).
        """

    @abstractmethod
    def clone_with(self, **overrides: Any) -> ModelAdapter:
        """Return a new, unfitted adapter with updated hyper-parameters.

        The original adapter is not modified.

        Args:
            **overrides: Hyper-parameter overrides.  Keys must be in the
               allow-listed parameter set for this adapter's algorithm.

        Returns:
            A new unfitted ``ModelAdapter`` with the overrides applied.

        Raises:
            InvalidModelSpec: If any override key/value is invalid.
        """

    @abstractmethod
    def capacity_grid(self) -> list[dict[str, Any]]:
        """Return the capacity-search grid for H1 (overfitting hypothesis).

        Returns:
            List of parameter dicts, each suitable for ``clone_with(**d)``.
        """

    @abstractmethod
    def supports_sample_weight(self) -> bool:
        """Return whether this adapter accepts ``sample_weight`` in ``fit``.

        Returns:
            ``True`` if the underlying estimator supports sample weights.
        """

    @abstractmethod
    def complexity(self) -> float:
        """Return a scalar measure of model complexity.

        Higher values indicate more complex (higher-variance) models that are
        more prone to overfitting.  Used as a diagnostic evidence value.

        Returns:
            Non-negative float.
        """
