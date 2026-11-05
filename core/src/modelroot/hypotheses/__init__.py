"""Hypotheses package (M9)."""

from __future__ import annotations

from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.hypotheses.registry import (
    clear_registry,
    get_registry,
    register_hypothesis,
)

__all__ = [
    "Applicability",
    "Hypothesis",
    "HypothesisContext",
    "clear_registry",
    "get_registry",
    "register_hypothesis",
]
