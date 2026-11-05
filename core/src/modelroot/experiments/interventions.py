"""Intervention registry (M9).

Interventions are pure functions registered by name.  The runner looks up
the callable by name from ``ExperimentSpec.intervention``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

InterventionFn = Callable[[Any, dict[str, Any]], Any]

_INTERVENTIONS: dict[str, InterventionFn] = {}


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
