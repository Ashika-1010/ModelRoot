"""Tests for M7 effect classification (T7-01 to T7-05)."""

from __future__ import annotations

import pytest

from modelroot.stats.effects import EffectClass, classify_effect


# T7-01: enum values
def test_effect_class_values() -> None:
    assert EffectClass.practically_significant.value == "practically_significant"
    assert EffectClass.significant_but_small.value == "significant_but_small"
    assert EffectClass.not_significant.value == "not_significant"
    assert EffectClass.negligible.value == "negligible"


# T7-02: negligible — entire CI below delta_min
def test_negligible_when_ci_high_below_delta_min() -> None:
    # ci_high = 0.01 < delta_min = 0.02
    assert classify_effect(0.005, -0.005, 0.01, 0.02) == EffectClass.negligible


# T7-03: practically_significant — ci_low > 0 AND delta >= delta_min
def test_practically_significant() -> None:
    assert classify_effect(0.05, 0.01, 0.09, 0.02) == EffectClass.practically_significant


# T7-04: significant_but_small — ci_low > 0 AND delta < delta_min
def test_significant_but_small() -> None:
    # ci_high >= delta_min (0.025 > 0.02), ci_low > 0, but delta < delta_min
    assert classify_effect(0.01, 0.001, 0.025, 0.02) == EffectClass.significant_but_small


# T7-05: not_significant — CI includes 0
def test_not_significant() -> None:
    assert classify_effect(0.01, -0.01, 0.03, 0.02) == EffectClass.not_significant


# Boundary: delta == delta_min is practically_significant if ci_low > 0
def test_delta_equals_delta_min() -> None:
    assert classify_effect(0.02, 0.001, 0.039, 0.02) == EffectClass.practically_significant


# Boundary: ci_high == delta_min → negligible (not <, but at boundary)
def test_ci_high_exactly_delta_min() -> None:
    # ci_high == 0.02 == delta_min → NOT < delta_min, so NOT negligible
    result = classify_effect(0.01, 0.0, 0.02, 0.02)
    assert result != EffectClass.negligible


# Negative delta, CI includes 0
def test_negative_delta_not_significant() -> None:
    assert classify_effect(-0.01, -0.05, 0.03, 0.02) == EffectClass.not_significant


# Validation
def test_delta_min_zero_raises() -> None:
    with pytest.raises(ValueError, match="delta_min"):
        classify_effect(0.05, 0.01, 0.09, 0.0)


def test_ci_inverted_raises() -> None:
    with pytest.raises(ValueError, match="ci_low"):
        classify_effect(0.05, 0.09, 0.01, 0.02)


def test_bootstrap_error_is_value_error() -> None:
    """BootstrapError must be a subclass of ValueError (regression guard)."""
    from modelroot.stats.bootstrap import BootstrapError

    assert issubclass(BootstrapError, ValueError)
