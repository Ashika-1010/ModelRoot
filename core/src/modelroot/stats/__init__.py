"""Statistics package: bootstrap, effects, repeated subsampling, NB, power (M7/M8/M15)."""

from __future__ import annotations

from modelroot.stats.bootstrap import (
    BootstrapError,
    BootstrapResult,
    CIResult,
    bootstrap_ci,
    paired_bootstrap,
    two_sample_bootstrap_diff,
)
from modelroot.stats.drift import (
    DomainClassifierResult,
    DriftReport,
    UnivariateDriftResult,
    compute_domain_classifier,
    compute_drift_report,
    compute_univariate_drift,
)
from modelroot.stats.effects import EffectClass, classify_effect
from modelroot.stats.multiple_testing import (
    adjust_p_values,
    benjamini_hochberg,
    benjamini_yekutieli,
)
from modelroot.stats.nadeau_bengio import NBTestResult, corrected_resampled_ttest, nb_to_contract
from modelroot.stats.permutation import PermutationResult, permutation_test_two_sample
from modelroot.stats.power import GateResult, is_negligible, minimum_detectable_effect, power_gate
from modelroot.stats.repeated import RepeatedResult, consistency, repeated_subsample_deltas
from modelroot.stats.subset_bootstrap import paired_subset_bootstrap

__all__ = [
    "BootstrapError",
    "BootstrapResult",
    "CIResult",
    "DomainClassifierResult",
    "DriftReport",
    "EffectClass",
    "GateResult",
    "NBTestResult",
    "PermutationResult",
    "RepeatedResult",
    "UnivariateDriftResult",
    "adjust_p_values",
    "benjamini_hochberg",
    "benjamini_yekutieli",
    "bootstrap_ci",
    "classify_effect",
    "compute_domain_classifier",
    "compute_drift_report",
    "compute_univariate_drift",
    "consistency",
    "corrected_resampled_ttest",
    "is_negligible",
    "minimum_detectable_effect",
    "nb_to_contract",
    "paired_bootstrap",
    "paired_subset_bootstrap",
    "permutation_test_two_sample",
    "power_gate",
    "repeated_subsample_deltas",
    "two_sample_bootstrap_diff",
]
