"""Tests for M6 Thresholds (T6-01 to T6-03)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelroot.config.thresholds import Thresholds


def test_thresholds_defaults() -> None:
    """T6-01: Default values match spec."""
    t = Thresholds()
    assert t.headroom == pytest.approx(0.03)
    assert t.gap == pytest.approx(0.10)
    assert t.current_drop == pytest.approx(0.03)


def test_thresholds_frozen() -> None:
    """T6-02: Thresholds is immutable after construction."""
    t = Thresholds()
    with pytest.raises((ValidationError, TypeError)):
        t.headroom = 0.05  # type: ignore[misc]


def test_thresholds_extra_field_rejected() -> None:
    """T6-03: Extra fields are rejected."""
    with pytest.raises(ValidationError):
        Thresholds(headroom=0.03, gap=0.10, current_drop=0.03, unknown=1)  # type: ignore[call-arg]


def test_thresholds_custom_values() -> None:
    t = Thresholds(headroom=0.05, gap=0.15, current_drop=0.05)
    assert t.headroom == 0.05
    assert t.gap == 0.15
