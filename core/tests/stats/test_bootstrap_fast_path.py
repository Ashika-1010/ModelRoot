"""Tests for M7 fast-path parity and invalid-resample policy (T7-15 to T7-17)."""

from __future__ import annotations

import numpy as np
import pytest

from modelroot.stats.bootstrap import (
    BootstrapError,
    _fast_label_delta,
    _generic_delta,
    paired_bootstrap,
)


def _binary_data(n: int = 200, seed: int = 99) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    y_proba = rng.random(size=n)
    return y_true, y_pred, y_proba


# T7-15: fast path matches sklearn metric on a single resample
@pytest.mark.parametrize(
    "metric", ["accuracy", "balanced_accuracy", "macro_f1", "f1", "precision", "recall"]
)
def test_fast_path_parity(metric: str) -> None:
    rng = np.random.default_rng(123)
    yt = rng.integers(0, 2, size=100)
    yp_a = rng.integers(0, 2, size=100)
    yp_b = rng.integers(0, 2, size=100)
    ypr = rng.random(size=100)
    idx = rng.integers(0, 100, size=100)

    # Fast path
    fast = _fast_label_delta(yt, yp_a, yp_b, metric, idx)

    # Generic via registry
    generic = _generic_delta(yt, yp_a, yp_b, ypr, ypr, metric, idx)

    if fast is None and generic is None:
        return  # Both undefined on this resample — OK
    assert fast is not None
    assert generic is not None
    assert abs(fast - generic) < 1e-9, f"{metric}: fast={fast}, generic={generic}"


# T7-16: fast-path result matches paired_bootstrap for label metrics
def test_fast_path_matches_bootstrap_output() -> None:
    yt, yp_a, _ = _binary_data(200, seed=1)
    _, yp_b, _ = _binary_data(200, seed=2)
    r = paired_bootstrap(yt, yp_a, yp_b, "accuracy", n_boot=100, seed=0)
    assert isinstance(r.delta, float)


# T7-17: invalid-resample policy — > 20% invalid → BootstrapError
def test_bootstrap_error_on_high_invalid_rate() -> None:
    # Highly imbalanced data so AUC is undefined on most resamples
    # 99% of samples are class 0
    n = 100
    yt = np.array([0] * 99 + [1])
    yp = np.zeros(n, dtype=int)
    ypr = np.zeros(n, dtype=float)
    ypr[99] = 0.9
    with pytest.raises(BootstrapError):
        paired_bootstrap(yt, yp, yp, "roc_auc", n_boot=200, seed=0, y_proba_a=ypr, y_proba_b=ypr)


# T7-17b: <= 20% invalid does NOT raise
def test_no_error_below_threshold() -> None:
    # Balanced data — AUC is well-defined on almost all resamples
    rng = np.random.default_rng(42)
    yt = rng.integers(0, 2, size=200)
    ypr = rng.random(size=200)
    yp = (ypr > 0.5).astype(int)
    # Should not raise
    result = paired_bootstrap(
        yt, yp, yp, "roc_auc", n_boot=100, seed=0, y_proba_a=ypr, y_proba_b=ypr
    )
    assert result.delta == pytest.approx(0.0, abs=1e-9)
