"""Statistics package: bootstrap confidence intervals, effect classification (M7)."""

from __future__ import annotations

from modelroot.stats.bootstrap import (
    BootstrapError,
    BootstrapResult,
    CIResult,
    bootstrap_ci,
    paired_bootstrap,
    two_sample_bootstrap_diff,
)
from modelroot.stats.effects import EffectClass, classify_effect

__all__ = [
    "BootstrapError",
    "BootstrapResult",
    "CIResult",
    "EffectClass",
    "bootstrap_ci",
    "classify_effect",
    "paired_bootstrap",
    "two_sample_bootstrap_diff",
]
