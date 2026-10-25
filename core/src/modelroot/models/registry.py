"""Model registry: maps ModelSpec → ModelAdapter (M4).

``get_model`` is the single entry point used by the rest of the codebase.
``list_model_specs`` returns a catalogue for the API/CLI.
"""

from __future__ import annotations

from typing import Any, Protocol

from modelroot.models.adapter import ModelAdapter
from modelroot.models.sklearn_adapters import (
    DecisionTreeAdapter,
    HistGradientBoostingAdapter,
    LogisticRegressionAdapter,
    RandomForestAdapter,
)
from modelroot.schemas.common import Algorithm
from modelroot.schemas.config import ModelSpec


class InvalidModelSpec(ValueError):
    """Raised when a ModelSpec contains unknown or out-of-range parameters."""

    def __init__(self, message: str) -> None:
        """Initialise with a human-readable message.

        Args:
            message: Description of the invalid spec.
        """
        super().__init__(message)


# Protocol for adapter classes that have DEFAULT_PARAMS and PARAM_SCHEMA class attrs.
# ``ModelAdapter`` (the ABC) doesn't declare these; only concrete subclasses do.


class _AdapterClassProtocol(Protocol):
    """Structural type for adapter classes with class-level defaults."""

    DEFAULT_PARAMS: dict[str, Any]
    PARAM_SCHEMA: dict[str, dict[str, Any]]

    def __call__(
        self,
        params: dict[str, Any],
        column_types: dict[str, str],
        seed: int,
    ) -> ModelAdapter: ...


# Mapping from Algorithm enum to adapter class
_REGISTRY: dict[Algorithm, _AdapterClassProtocol] = {
    Algorithm.logistic_regression: LogisticRegressionAdapter,
    Algorithm.decision_tree: DecisionTreeAdapter,
    Algorithm.random_forest: RandomForestAdapter,
    Algorithm.hist_gradient_boosting: HistGradientBoostingAdapter,
}


def get_model(
    spec: ModelSpec,
    column_types: dict[str, str],
    seed: int,
) -> ModelAdapter:
    """Instantiate a ``ModelAdapter`` from a ``ModelSpec``.

    Args:
        spec: Algorithm name and hyper-parameters.
        column_types: Mapping from feature column name to ``"numeric"`` or
            ``"categorical"``.  Passed to the preprocessor builder.
        seed: Integer random seed used to set ``random_state`` on the estimator.

    Returns:
        An unfitted ``ModelAdapter``.

    Raises:
        InvalidModelSpec: If ``spec.algorithm`` is not registered or any
            ``spec.params`` key/value violates the allow-list.
    """
    adapter_cls = _REGISTRY.get(spec.algorithm)
    if adapter_cls is None:
        raise InvalidModelSpec(
            f"Unknown algorithm: {spec.algorithm!r}. "
            f"Supported: {sorted(a.value for a in _REGISTRY)}"
        )
    return adapter_cls(params=dict(spec.params), column_types=column_types, seed=seed)


def list_model_specs() -> list[dict[str, Any]]:
    """Return a list of algorithm descriptions with defaults and param schemas.

    Returns:
        List of dicts, one per algorithm, with keys:
        ``algorithm``, ``defaults``, ``param_schema``.
    """
    result = []
    for algo, adapter_cls in _REGISTRY.items():
        result.append(
            {
                "algorithm": algo.value,
                "defaults": adapter_cls.DEFAULT_PARAMS,
                "param_schema": adapter_cls.PARAM_SCHEMA,
            }
        )
    return result
