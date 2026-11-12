"""Intervention registry and outcome type (M9/M11).

Interventions are pure functions registered by name.  The runner looks up
the callable by name from ``ExperimentSpec.intervention``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

InterventionFn = Callable[[Any, dict[str, Any]], Any]

_INTERVENTIONS: dict[str, InterventionFn] = {}


@dataclass
class InterventionOutcome:
    """Result returned by an intervention function.

    Attributes:
        holdout_pred: Predicted labels on the holdout set (1-D int array).
        holdout_proba: P(class=1) scores on the holdout set (1-D float array).
        fitted_adapter: The fitted model adapter after the intervention.
        factory: Zero-arg callable that returns a fresh *unfitted* adapter
            with the selected hyperparameters fixed (used for repeated subsampling).
        current_pred: Optional predictions on the current set.
        current_proba: Optional P(class=1) scores on the current set.
        notes: Free-text notes (selected params, scores, etc.).
    """

    holdout_pred: np.ndarray
    holdout_proba: np.ndarray
    fitted_adapter: Any
    factory: Callable[[], Any]
    current_pred: np.ndarray | None = None
    current_proba: np.ndarray | None = None
    notes: list[str] = field(default_factory=list)


def intervention(name: str) -> Callable[[InterventionFn], InterventionFn]:
    """Decorator to register an intervention function by *name*.

    Args:
        name: Stable string identifier for this intervention.

    Returns:
        Decorator that registers the function and returns it unchanged.

    Raises:
        ValueError: If *name* is already registered.
    """

    def decorator(fn: InterventionFn) -> InterventionFn:
        if name in _INTERVENTIONS:
            raise ValueError(f"Intervention {name!r} is already registered.")
        _INTERVENTIONS[name] = fn
        return fn

    return decorator


def get_intervention(name: str) -> InterventionFn:
    """Look up a registered intervention function.

    Args:
        name: Intervention identifier.

    Returns:
        The registered callable.

    Raises:
        KeyError: If *name* is not registered.
    """
    try:
        return _INTERVENTIONS[name]
    except KeyError:
        raise KeyError(
            f"Intervention {name!r} is not registered. Available: {sorted(_INTERVENTIONS)}"
        ) from None


def list_interventions() -> list[str]:
    """Return all registered intervention names in sorted order."""
    return sorted(_INTERVENTIONS)
