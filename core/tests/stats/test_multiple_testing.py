"""Tests for multiple testing correction procedures (M17: BH and BY)."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.stats import false_discovery_control

from modelroot.stats.multiple_testing import (
    adjust_p_values,
    benjamini_hochberg,
    benjamini_yekutieli,
)


class TestBenjaminiHochberg:
    """Tests for Benjamini-Hochberg (BH) procedure."""

    def test_empty_input(self) -> None:
        assert benjamini_hochberg([]) == []

    def test_single_value(self) -> None:
        assert benjamini_hochberg([0.04]) == [0.04]
        assert benjamini_hochberg([0.0]) == [0.0]
        assert benjamini_hochberg([1.0]) == [1.0]

    def test_matches_scipy_exact(self) -> None:
        # Fixed benchmark set of p-values
        p_vals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.900]
        ours = benjamini_hochberg(p_vals)
        scipy_res = false_discovery_control(p_vals, method="bh")
        np.testing.assert_allclose(ours, scipy_res, rtol=1e-12, atol=1e-12)

    def test_random_samples_match_scipy(self) -> None:
        rng = np.random.default_rng(2026)
        for _ in range(10):
            p_vals = rng.uniform(0.0, 1.0, size=50).tolist()
            ours = benjamini_hochberg(p_vals)
            scipy_res = false_discovery_control(p_vals, method="bh")
            np.testing.assert_allclose(ours, scipy_res, rtol=1e-12, atol=1e-12)

    def test_original_order_preserved(self) -> None:
        # Unsorted inputs
        p_vals = [0.5, 0.01, 0.04, 0.02, 0.9]
        q_vals = benjamini_hochberg(p_vals)
        # Smallest p-value (index 1: 0.01) must have smallest q-value
        assert q_vals[1] < q_vals[2] < q_vals[0] < q_vals[4]

    def test_handling_ties_and_repeated_values(self) -> None:
        p_vals = [0.01, 0.04, 0.04, 0.04, 0.10]
        ours = benjamini_hochberg(p_vals)
        scipy_res = false_discovery_control(p_vals, method="bh")
        np.testing.assert_allclose(ours, scipy_res, rtol=1e-12, atol=1e-12)
        # Tied inputs must receive identical adjusted values
        assert ours[1] == ours[2] == ours[3]

    def test_monotonicity_in_sorted_order(self) -> None:
        rng = np.random.default_rng(42)
        p_vals = sorted(rng.uniform(0.0, 1.0, size=100).tolist())
        q_vals = benjamini_hochberg(p_vals)
        # In sorted p order, q-values must be non-decreasing
        for i in range(len(q_vals) - 1):
            assert q_vals[i] <= q_vals[i + 1] + 1e-14

    def test_clipping_to_one(self) -> None:
        p_vals = [0.8, 0.9, 0.95]
        q_vals = benjamini_hochberg(p_vals)
        assert all(0.0 <= q <= 1.0 for q in q_vals)
        assert q_vals == [0.95, 0.95, 0.95]


class TestBenjaminiYekutieli:
    """Tests for Benjamini-Yekutieli (BY) procedure."""

    def test_empty_input(self) -> None:
        assert benjamini_yekutieli([]) == []

    def test_single_value(self) -> None:
        # For m=1, c(m)=1, so BY == BH
        assert benjamini_yekutieli([0.05]) == [0.05]

    def test_matches_scipy_exact(self) -> None:
        p_vals = [0.001, 0.008, 0.039, 0.041, 0.042, 0.060, 0.074, 0.205, 0.212, 0.900]
        ours = benjamini_yekutieli(p_vals)
        scipy_res = false_discovery_control(p_vals, method="by")
        np.testing.assert_allclose(ours, scipy_res, rtol=1e-12, atol=1e-12)

    def test_random_samples_match_scipy(self) -> None:
        rng = np.random.default_rng(2026)
        for _ in range(10):
            p_vals = rng.uniform(0.0, 1.0, size=50).tolist()
            ours = benjamini_yekutieli(p_vals)
            scipy_res = false_discovery_control(p_vals, method="by")
            np.testing.assert_allclose(ours, scipy_res, rtol=1e-12, atol=1e-12)

    def test_by_at_least_as_conservative_as_bh(self) -> None:
        rng = np.random.default_rng(1234)
        for size in [5, 20, 100]:
            p_vals = rng.uniform(0.0, 1.0, size=size).tolist()
            bh_q = benjamini_hochberg(p_vals)
            by_q = benjamini_yekutieli(p_vals)
            for q_bh, q_by in zip(bh_q, by_q, strict=True):
                assert q_by >= q_bh - 1e-14, f"BY ({q_by}) was less conservative than BH ({q_bh})"

    def test_by_ties_handling(self) -> None:
        p_vals = [0.02, 0.05, 0.05, 0.05, 0.2]
        ours = benjamini_yekutieli(p_vals)
        scipy_res = false_discovery_control(p_vals, method="by")
        np.testing.assert_allclose(ours, scipy_res, rtol=1e-12, atol=1e-12)
        assert ours[1] == ours[2] == ours[3]


class TestValidationAndWrapper:
    """Tests for input validation and the adjust_p_values wrapper."""

    def test_invalid_p_value_ranges(self) -> None:
        with pytest.raises(ValueError, match="interval \\[0, 1\\]"):
            benjamini_hochberg([-0.01, 0.5])

        with pytest.raises(ValueError, match="interval \\[0, 1\\]"):
            benjamini_yekutieli([0.5, 1.05])

    def test_nans_and_infs(self) -> None:
        with pytest.raises(ValueError, match="NaN"):
            benjamini_hochberg([0.1, float("nan")])

        with pytest.raises(ValueError, match="Inf"):
            benjamini_yekutieli([0.1, float("inf")])

    def test_multidimensional_input(self) -> None:
        with pytest.raises(ValueError, match="1D"):
            benjamini_hochberg([[0.1, 0.2], [0.3, 0.4]])  # type: ignore[arg-type]

    def test_wrapper_dispatch(self) -> None:
        p_vals = [0.01, 0.05, 0.1]
        assert adjust_p_values(p_vals, method="bh") == benjamini_hochberg(p_vals)
        assert adjust_p_values(p_vals, method="by") == benjamini_yekutieli(p_vals)
        assert adjust_p_values(p_vals, method="BY") == benjamini_yekutieli(p_vals)

        with pytest.raises(ValueError, match="Unknown multiple testing method"):
            adjust_p_values(p_vals, method="bonferroni")
