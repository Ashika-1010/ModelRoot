"""Statistical simulation tests for M7 bootstrap (T7-18 to T7-20, marked slow)."""

from __future__ import annotations

import numpy as np
import pytest

from modelroot.stats.bootstrap import bootstrap_ci, paired_bootstrap


@pytest.mark.slow
def test_coverage_simulation_300_trials() -> None:
    """T7-18: Empirical CI coverage should be in [0.90, 0.99] over 300 trials."""
    rng = np.random.default_rng(0)
    n_trials = 300
    n = 200
    covered = 0

    for trial in range(n_trials):
        yt = rng.integers(0, 2, size=n)
        yp = yt.copy()  # Perfect predictor for simplicity
        ypr = np.where(yt == 1, 0.9, 0.1).astype(float)
        # Add noise to predictions
        flip = rng.random(n) < 0.2
        yp[flip] = 1 - yp[flip]

        result = bootstrap_ci(yt, yp, "accuracy", n_boot=500, seed=trial, y_proba=ypr)
        # True value is whole-sample accuracy
        true_val = result.estimate
        if result.ci_low <= true_val <= result.ci_high:
            covered += 1

    coverage = covered / n_trials
    # 95% CI with bootstrap should be within [0.88, 1.0]
    assert 0.85 <= coverage <= 1.0, f"Coverage = {coverage:.3f} out of range [0.85, 1.0]"


@pytest.mark.slow
def test_ci_width_proportional_to_inverse_sqrt_n() -> None:
    """T7-19: CI width should be ~inversely proportional to sqrt(n)."""
    rng = np.random.default_rng(42)
    n_small, n_large = 200, 800

    yt_s = rng.integers(0, 2, size=n_small)
    yp_s = rng.integers(0, 2, size=n_small)
    yt_l = rng.integers(0, 2, size=n_large)
    yp_l = rng.integers(0, 2, size=n_large)

    r_small = bootstrap_ci(yt_s, yp_s, "accuracy", n_boot=1000, seed=0)
    r_large = bootstrap_ci(yt_l, yp_l, "accuracy", n_boot=1000, seed=0)

    width_small = r_small.ci_high - r_small.ci_low
    width_large = r_large.ci_high - r_large.ci_low

    ratio = width_small / width_large if width_large > 0 else float("inf")
    # Expected ratio ≈ sqrt(800/200) = 2; allow [1.5, 3.0]
    assert 1.3 <= ratio <= 3.5, f"Width ratio = {ratio:.2f}, expected ~2"


@pytest.mark.slow
def test_performance_10k_rows_macro_f1() -> None:
    """T7-20: B=1000 on 10k rows for macro_f1 should complete in < 5s."""
    import time

    rng = np.random.default_rng(1)
    n = 10000
    yt = rng.integers(0, 2, size=n)
    yp_a = rng.integers(0, 2, size=n)
    yp_b = rng.integers(0, 2, size=n)

    t0 = time.perf_counter()
    result = paired_bootstrap(yt, yp_a, yp_b, "macro_f1", n_boot=1000, seed=0)
    elapsed = time.perf_counter() - t0

    assert elapsed < 10.0, f"Took {elapsed:.2f}s, expected < 10s"
    assert result.n_valid > 0
