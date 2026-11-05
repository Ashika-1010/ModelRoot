"""Statistics package: bootstrap, effects, repeated subsampling, NB, power (M7/M8)."""

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
from modelroot.stats.nadeau_bengio import NBTestResult, corrected_resampled_ttest, nb_to_contract
from modelroot.stats.power import GateResult, is_negligible, minimum_detectable_effect, power_gate
from modelroot.stats.repeated import RepeatedResult, consistency, repeated_subsample_deltas

__all__ = [
    "BootstrapError",
    "BootstrapResult",
    "CIResult",
    "EffectClass",
    "GateResult",
    "NBTestResult",
    "RepeatedResult",
    "bootstrap_ci",
    "classify_effect",
    "consistency",
    "corrected_resampled_ttest",
    "is_negligible",
    "minimum_detectable_effect",
    "nb_to_contract",
    "paired_bootstrap",
    "power_gate",
    "repeated_subsample_deltas",
    "two_sample_bootstrap_diff",
]
