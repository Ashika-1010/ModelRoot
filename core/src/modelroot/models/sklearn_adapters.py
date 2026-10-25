"""Concrete sklearn adapter implementations (M4).

Each adapter:
  1. Validates parameters against an allow-list.
  2. Builds an sklearn Pipeline(preprocessor, estimator).
  3. Exposes the ModelAdapter protocol.

Allow-listed parameters per algorithm (from Â§9.2 M4 spec):
  LR : C, penalty, max_iter, class_weight
  DT : max_depth, min_samples_leaf, min_samples_split, class_weight
  RF : n_estimators, max_depth, min_samples_leaf, max_features, class_weight
  HGB: max_depth, learning_rate, l2_regularization, max_iter, early_stopping
"""

from __future__ import annotations

from typing import Any, ClassVar

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier

from modelroot.data.preprocessing import build_preprocessor
from modelroot.models.adapter import ModelAdapter

# ---------------------------------------------------------------------------
# Parameter validation helpers
# ---------------------------------------------------------------------------


def _validate_params(
    params: dict[str, Any],
    schema: dict[str, dict[str, Any]],
    algo_name: str,
) -> None:
    """Validate *params* against *schema*.

    Args:
        params: User-supplied hyper-parameters.
        schema: Mapping from param name to its constraints dict.
        algo_name: Algorithm name for error messages.

    Raises:
        modelroot.models.registry.InvalidModelSpec: On any violation.
    """
    from modelroot.models.registry import InvalidModelSpec

    unknown = set(params) - set(schema)
    if unknown:
        raise InvalidModelSpec(
            f"[{algo_name}] Unknown parameter(s): {sorted(unknown)}. Allowed: {sorted(schema)}"
        )
    for name, value in params.items():
        constraints = schema[name]
        allowed_types = constraints.get("type")
        nullable = constraints.get("nullable", False)
        if (
            allowed_types
            and not isinstance(value, allowed_types)
            and (value is not None or not nullable)
        ):
            raise InvalidModelSpec(
                f"[{algo_name}] Parameter '{name}' must be of type "
                f"{allowed_types}, got {type(value).__name__!r}."
            )
        allowed_values = constraints.get("values")
        if allowed_values is not None and value not in allowed_values:
            raise InvalidModelSpec(
                f"[{algo_name}] Parameter '{name}' must be one of {allowed_values}; got {value!r}."
            )
        min_val = constraints.get("min")
        if min_val is not None and value is not None and value < min_val:
            raise InvalidModelSpec(
                f"[{algo_name}] Parameter '{name}' must be >= {min_val}; got {value!r}."
            )
        max_val = constraints.get("max")
        if max_val is not None and value is not None and value > max_val:
            raise InvalidModelSpec(
                f"[{algo_name}] Parameter '{name}' must be <= {max_val}; got {value!r}."
            )


def _reject_injection(params: dict[str, Any], algo_name: str) -> None:
    """Reject params whose string values look like injection attempts.

    This is a defence-in-depth guard; the primary protection is the explicit
    type + value allow-list in ``_validate_params``.

    Args:
        params: User-supplied parameters.
        algo_name: Used for error messages.
    """
    from modelroot.models.registry import InvalidModelSpec

    dangerous_patterns = ["import", "__", "eval(", "exec(", "os.", "sys."]
    for name, value in params.items():
        if isinstance(value, str):
            for pat in dangerous_patterns:
                if pat in value:
                    raise InvalidModelSpec(
                        f"[{algo_name}] Parameter '{name}' contains a potentially "
                        f"unsafe string: {value!r}."
                    )


# ---------------------------------------------------------------------------
# Base adapter
# ---------------------------------------------------------------------------


class _BaseSklearnAdapter(ModelAdapter):
    """Shared implementation for all sklearn-backed adapters."""

    ALGO_NAME: str = "base"
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {}
    PARAM_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {}
    _USE_SCALE: bool = False

    def __init__(
        self,
        params: dict[str, Any],
        column_types: dict[str, str],
        seed: int,
    ) -> None:
        """Initialise with validated params, column types, and seed.

        Args:
            params: Hyper-parameters (must pass validation).
            column_types: Feature column type mapping.
            seed: Random seed for the estimator.
        """
        _reject_injection(params, self.ALGO_NAME)
        _validate_params(params, self.PARAM_SCHEMA, self.ALGO_NAME)
        self._params = {**self.DEFAULT_PARAMS, **params}
        self._column_types = column_types
        self._seed = seed
        self._pipeline: Pipeline | None = None

    @property
    def params(self) -> dict[str, Any]:
        """Current hyper-parameters (merged defaults + overrides)."""
        return dict(self._params)

    def _build_estimator(self) -> object:
        raise NotImplementedError

    def _build_pipeline(self) -> Pipeline:
        preprocessor = build_preprocessor(self._column_types, scale=self._USE_SCALE)
        estimator = self._build_estimator()
        return Pipeline(steps=[("preprocessor", preprocessor), ("estimator", estimator)])

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sample_weight: np.ndarray | None = None,
    ) -> _BaseSklearnAdapter:
        """Fit the adapter pipeline on training data."""
        self._pipeline = self._build_pipeline()
        fit_params: dict[str, Any] = {}
        if sample_weight is not None and self.supports_sample_weight():
            fit_params["estimator__sample_weight"] = sample_weight
        self._pipeline.fit(X, y, **fit_params)
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict class labels (0 or 1)."""
        if self._pipeline is None:
            raise RuntimeError("Model must be fitted before calling predict()")
        return np.asarray(self._pipeline.predict(X))

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return (n, 2) class probability array."""
        if self._pipeline is None:
            raise RuntimeError("Model must be fitted before calling predict_proba()")
        proba = self._pipeline.predict_proba(X)
        if proba.ndim == 1 or proba.shape[1] != 2:
            raise ValueError(f"Expected (n, 2) proba array, got shape {proba.shape}")
        return np.asarray(proba, dtype=float)

    def clone_with(self, **overrides: Any) -> _BaseSklearnAdapter:
        """Return a new unfitted adapter with overridden parameters."""
        merged = {**self._params, **overrides}
        return self.__class__(
            params=merged,
            column_types=self._column_types,
            seed=self._seed,
        )

    def supports_sample_weight(self) -> bool:
        return False

    def complexity(self) -> float:
        return 1.0


# ---------------------------------------------------------------------------
# LogisticRegression
# ---------------------------------------------------------------------------


class LogisticRegressionAdapter(_BaseSklearnAdapter):
    """Logistic Regression adapter with StandardScaler preprocessing."""

    ALGO_NAME = "logistic_regression"
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "C": 1.0,
        "penalty": "l2",
        "max_iter": 1000,
        "class_weight": None,
    }
    PARAM_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "C": {"type": (int, float), "min": 1e-6},
        "penalty": {"type": str, "values": ["l2", "none"]},
        "max_iter": {"type": int, "min": 1},
        "class_weight": {"nullable": True, "values": [None, "balanced"]},
    }
    _USE_SCALE = True

    def _build_estimator(self) -> LogisticRegression:
        # sklearn 1.8 deprecated the ``penalty`` keyword argument; the new
        # API uses ``l1_ratio`` (0.0 = L2, 1.0 = L1) and ``C=inf`` for no
        # regularization.  We translate the public ``penalty`` parameter to
        # the new API so the estimator receives the correct regularization
        # without triggering a FutureWarning.
        penalty = self._params["penalty"]
        if penalty == "none":
            # No regularization: C=inf disables it.
            return LogisticRegression(
                C=float("inf"),
                l1_ratio=0.0,
                max_iter=int(self._params["max_iter"]),
                class_weight=self._params["class_weight"],
                random_state=self._seed,
                solver="lbfgs",
            )
        # penalty == "l2" (the only other allowed value)
        return LogisticRegression(
            C=float(self._params["C"]),
            l1_ratio=0.0,
            max_iter=int(self._params["max_iter"]),
            class_weight=self._params["class_weight"],
            random_state=self._seed,
            solver="lbfgs",
        )

    def supports_sample_weight(self) -> bool:
        return False

    def complexity(self) -> float:
        """Higher C = less regularization = more complex."""
        return float(self._params["C"])

    def capacity_grid(self) -> list[dict[str, Any]]:
        """LR capacity grid from Â§5-H1: C in {1, 0.1, 0.01, 0.001}."""
        return [{"C": c} for c in [1.0, 0.1, 0.01, 0.001]]


# ---------------------------------------------------------------------------
# DecisionTree
# ---------------------------------------------------------------------------


class DecisionTreeAdapter(_BaseSklearnAdapter):
    """Decision Tree adapter (no scaling)."""

    ALGO_NAME = "decision_tree"
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "max_depth": None,
        "min_samples_leaf": 1,
        "min_samples_split": 2,
        "class_weight": None,
    }
    PARAM_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "max_depth": {"nullable": True, "type": (int,), "min": 1},
        "min_samples_leaf": {"type": int, "min": 1},
        "min_samples_split": {"type": int, "min": 2},
        "class_weight": {"nullable": True, "values": [None, "balanced"]},
    }
    _USE_SCALE = False

    def _build_estimator(self) -> DecisionTreeClassifier:
        return DecisionTreeClassifier(
            max_depth=self._params["max_depth"],
            min_samples_leaf=int(self._params["min_samples_leaf"]),
            min_samples_split=int(self._params["min_samples_split"]),
            class_weight=self._params["class_weight"],
            random_state=self._seed,
        )

    def supports_sample_weight(self) -> bool:
        return True

    def complexity(self) -> float:
        """Unbounded depth = maximum complexity."""
        d = self._params["max_depth"]
        if d is None:
            return 100.0
        return float(d)

    def capacity_grid(self) -> list[dict[str, Any]]:
        """DT capacity grid from Â§5-H1: max_depth x min_samples_leaf."""
        return [{"max_depth": md, "min_samples_leaf": ml} for md in [3, 5, 8] for ml in [5, 20]]


# ---------------------------------------------------------------------------
# RandomForest
# ---------------------------------------------------------------------------


class RandomForestAdapter(_BaseSklearnAdapter):
    """Random Forest adapter (no scaling)."""

    ALGO_NAME = "random_forest"
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "class_weight": None,
    }
    PARAM_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "n_estimators": {"type": int, "min": 1, "max": 500},
        "max_depth": {"nullable": True, "type": (int,), "min": 1},
        "min_samples_leaf": {"type": int, "min": 1},
        "max_features": {"nullable": True, "values": [None, "sqrt", "log2", 0.3, 0.5]},
        "class_weight": {"nullable": True, "values": [None, "balanced"]},
    }
    _USE_SCALE = False

    def _build_estimator(self) -> RandomForestClassifier:
        return RandomForestClassifier(
            n_estimators=int(self._params["n_estimators"]),
            max_depth=self._params["max_depth"],
            min_samples_leaf=int(self._params["min_samples_leaf"]),
            max_features=self._params["max_features"],
            class_weight=self._params["class_weight"],
            random_state=self._seed,
            n_jobs=1,
        )

    def supports_sample_weight(self) -> bool:
        return True

    def complexity(self) -> float:
        """Complexity scales with depth and number of trees."""
        d = self._params["max_depth"]
        if d is None:
            return float(self._params["n_estimators"])
        return float(d) * float(self._params["n_estimators"]) / 100.0

    def capacity_grid(self) -> list[dict[str, Any]]:
        """RF capacity grid from Â§5-H1: min_samples_leaf x max_depth x max_features."""
        return [
            {"min_samples_leaf": ml, "max_depth": md, "max_features": mf}
            for ml in [5, 20]
            for md in [6, 10]
            for mf in ["sqrt", 0.3]
        ]


# ---------------------------------------------------------------------------
# HistGradientBoosting
# ---------------------------------------------------------------------------


class HistGradientBoostingAdapter(_BaseSklearnAdapter):
    """HistGradientBoosting adapter (no scaling; handles missing values natively)."""

    ALGO_NAME = "hist_gradient_boosting"
    DEFAULT_PARAMS: ClassVar[dict[str, Any]] = {
        "max_depth": None,
        "learning_rate": 0.1,
        "l2_regularization": 0.0,
        "max_iter": 100,
        "early_stopping": "auto",
    }
    PARAM_SCHEMA: ClassVar[dict[str, dict[str, Any]]] = {
        "max_depth": {"nullable": True, "type": (int,), "min": 1},
        "learning_rate": {"type": (int, float), "min": 1e-6, "max": 1.0},
        "l2_regularization": {"type": (int, float), "min": 0.0},
        "max_iter": {"type": int, "min": 1, "max": 500},
        "early_stopping": {"nullable": True, "values": ["auto", True, False, None]},
    }
    _USE_SCALE = False

    def _build_estimator(self) -> HistGradientBoostingClassifier:
        return HistGradientBoostingClassifier(
            max_depth=self._params["max_depth"],
            learning_rate=float(self._params["learning_rate"]),
            l2_regularization=float(self._params["l2_regularization"]),
            max_iter=int(self._params["max_iter"]),
            early_stopping=self._params["early_stopping"],
            random_state=self._seed,
        )

    def supports_sample_weight(self) -> bool:
        return True

    def complexity(self) -> float:
        """Complexity scales with max_iter and learning_rate."""
        return float(self._params["max_iter"]) * float(self._params["learning_rate"])

    def capacity_grid(self) -> list[dict[str, Any]]:
        """HGB capacity grid from Â§5-H1: max_depth x l2_regularization x learning_rate."""
        return [
            {
                "max_depth": md,
                "l2_regularization": l2,
                "learning_rate": lr,
                "early_stopping": True,
            }
            for md in [3, 6]
            for l2 in [0.0, 1.0]
            for lr in [0.1, 0.01]
        ]
