"""Tests for M7 bootstrap core: identity, antisymmetry, determinism (T7-06 to T7-12)."""

from __future__ import annotations

import numpy as np
import pytest

from modelroot.stats.bootstrap import (
    BootstrapResult,
    bootstrap_ci,
    paired_bootstrap,
    two_sample_bootstrap_diff,
)


def _binary_data(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    y_proba = rng.random(size=n)
    return y_true, y_pred, y_proba


# T7-06: identical predictions → delta = 0 and CI = [0, 0]
def test_identical_predictions_delta_zero() -> None:
    yt, yp, ypr = _binary_data(200)
    result = paired_bootstrap(
        yt, yp, yp, "macro_f1", n_boot=200, seed=0, y_proba_a=ypr, y_proba_b=ypr
    )
    assert result.delta == pytest.approx(0.0, abs=1e-12)
    assert result.ci_low == pytest.approx(0.0, abs=1e-9)
    assert result.ci_high == pytest.approx(0.0, abs=1e-9)


# T7-07: antisymmetry (swap a,b → delta negated, same seed)
def test_antisymmetry() -> None:
    yt, yp_a, ypr_a = _binary_data(200, seed=1)
    _, yp_b, ypr_b = _binary_data(200, seed=2)
    r_ab = paired_bootstrap(
        yt, yp_a, yp_b, "macro_f1", n_boot=200, seed=7, y_proba_a=ypr_a, y_proba_b=ypr_b
    )
    r_ba = paired_bootstrap(
        yt, yp_b, yp_a, "macro_f1", n_boot=200, seed=7, y_proba_a=ypr_b, y_proba_b=ypr_a
    )
    assert r_ab.delta == pytest.approx(-r_ba.delta, abs=1e-12)
    assert r_ab.ci_low == pytest.approx(-r_ba.ci_high, abs=1e-9)
    assert r_ab.ci_high == pytest.approx(-r_ba.ci_low, abs=1e-9)


# T7-08: determinism — same inputs → same result
def test_determinism() -> None:
    yt, yp_a, ypr_a = _binary_data(200, seed=1)
    _, yp_b, ypr_b = _binary_data(200, seed=2)
    r1 = paired_bootstrap(
        yt, yp_a, yp_b, "macro_f1", n_boot=200, seed=7, y_proba_a=ypr_a, y_proba_b=ypr_b
    )
    r2 = paired_bootstrap(
        yt, yp_a, yp_b, "macro_f1", n_boot=200, seed=7, y_proba_a=ypr_a, y_proba_b=ypr_b
    )
    assert r1.delta == r2.delta
    assert r1.ci_low == r2.ci_low
    assert r1.ci_high == r2.ci_high


# T7-09: delta == metric_b - metric_a
def test_delta_equals_b_minus_a() -> None:
    yt, yp_a, ypr_a = _binary_data(200, seed=1)
    _, yp_b, ypr_b = _binary_data(200, seed=2)
    result = paired_bootstrap(yt, yp_a, yp_b, "accuracy", n_boot=200, seed=0)
    from modelroot.assessment.metrics import metric_fn

    fn = metric_fn("accuracy")
    expected = fn(yt, yp_b, ypr_b) - fn(yt, yp_a, ypr_a)
    assert result.delta == pytest.approx(expected, abs=1e-12)


# T7-10: validation — n_boot < 100 raises
def test_n_boot_too_small_raises() -> None:
    yt, yp, ypr = _binary_data(50)
    with pytest.raises(ValueError, match="n_boot"):
        paired_bootstrap(yt, yp, yp, "accuracy", n_boot=99, seed=0)


# T7-11: validation — alpha out of range raises
def test_alpha_out_of_range_raises() -> None:
    yt, yp, ypr = _binary_data(50)
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap(yt, yp, yp, "accuracy", n_boot=100, seed=0, alpha=0.0)
    with pytest.raises(ValueError, match="alpha"):
        paired_bootstrap(yt, yp, yp, "accuracy", n_boot=100, seed=0, alpha=1.0)


# T7-12: BootstrapResult has expected fields
def test_bootstrap_result_fields() -> None:
    yt, yp, ypr = _binary_data(100)
    result = paired_bootstrap(yt, yp, yp, "accuracy", n_boot=100, seed=0)
    assert isinstance(result, BootstrapResult)
    assert result.n_boot == 100
    assert result.n_valid <= result.n_boot
    assert result.ci_low <= result.ci_high


# T7-13: bootstrap_ci returns estimate in [0,1] for accuracy
def test_bootstrap_ci_estimate() -> None:
    yt, yp, ypr = _binary_data(100)
    result = bootstrap_ci(yt, yp, "accuracy", n_boot=100, seed=0)
    assert 0.0 <= result.estimate <= 1.0
    assert result.ci_low <= result.estimate


# T7-14: two_sample_bootstrap_diff — different dataset sizes
def test_two_sample_different_sizes() -> None:
    yt_a, yp_a, _ = _binary_data(100, seed=1)
    yt_b, yp_b, _ = _binary_data(150, seed=2)
    result = two_sample_bootstrap_diff(yt_a, yp_a, yt_b, yp_b, "accuracy", n_boot=100, seed=0)
    assert isinstance(result, BootstrapResult)
    assert result.n_valid > 0
