"""Hypothesis registry (M9/M11).

Hypotheses are registered in order; ``diagnose`` iterates them in
registration order. The registry is module-level (global by default) but
an isolated registry can be created for testing.

``default_hypotheses()`` returns the canonical ordered tuple that includes
H1 first, followed by any explicitly registered hypotheses.
"""

from __future__ import annotations

from modelroot.hypotheses.base import Hypothesis

# Global hypothesis registry — in registration order (user-registered classes)
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

    This is the *user-registered* registry; it does not include H1 by default.
    Use :func:`default_hypotheses` for the full canonical list.

    Returns:
        List of registered hypothesis classes in registration order.
    """
    return list(_REGISTRY)


def clear_registry() -> None:
    """Remove all entries from the global registry (for testing only)."""
    _REGISTRY.clear()


def default_hypotheses() -> tuple[type[Hypothesis], ...]:
    """Return the canonical ordered tuple of hypotheses for ``diagnose()``.

    H1 (OverfittingHypothesis) is always first, followed by H2 (ImbalanceHypothesis),
    followed by any hypotheses in the global registry in registration order.

    Returns:
        Tuple of hypothesis *classes* in run order.
    """
    from modelroot.hypotheses.imbalance import ImbalanceHypothesis
    from modelroot.hypotheses.overfitting import OverfittingHypothesis

    seen: list[type[Hypothesis]] = [OverfittingHypothesis, ImbalanceHypothesis]
    for cls in _REGISTRY:
        if cls not in seen:
            seen.append(cls)
    return tuple(seen)
