"""Hypothesis registry (M9).

Hypotheses are registered in order; ``diagnose`` iterates them in
registration order. The registry is module-level (global by default) but
an isolated registry can be created for testing.
"""

from __future__ import annotations

from modelroot.hypotheses.base import Hypothesis

# Global hypothesis registry — in registration order
_REGISTRY: list[type[Hypothesis]] = []


def register_hypothesis(cls: type[Hypothesis]) -> type[Hypothesis]:
    """Register *cls* in the global hypothesis registry.

    Can be used as a decorator::

        @register_hypothesis
        class OverfittingHypothesis(Hypothesis):
            ...

    Args:
        cls: Hypothesis subclass to register.

    Returns:
        *cls* unchanged.
    """
    if cls not in _REGISTRY:
        _REGISTRY.append(cls)
    return cls


def get_registry() -> list[type[Hypothesis]]:
    """Return a shallow copy of the current global registry.

    Returns:
        List of registered hypothesis classes in registration order.
    """
    return list(_REGISTRY)


def clear_registry() -> None:
    """Remove all entries from the global registry (for testing only)."""
    _REGISTRY.clear()
