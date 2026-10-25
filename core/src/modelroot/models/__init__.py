"""Model adapters and registry for ModelRoot (M4)."""

from __future__ import annotations

from modelroot.models.adapter import ModelAdapter
from modelroot.models.registry import InvalidModelSpec, get_model, list_model_specs
from modelroot.models.sklearn_adapters import (
    DecisionTreeAdapter,
    HistGradientBoostingAdapter,
    LogisticRegressionAdapter,
    RandomForestAdapter,
)

__all__ = [
    "DecisionTreeAdapter",
    "HistGradientBoostingAdapter",
    "InvalidModelSpec",
    "LogisticRegressionAdapter",
    "ModelAdapter",
    "RandomForestAdapter",
    "get_model",
    "list_model_specs",
]
