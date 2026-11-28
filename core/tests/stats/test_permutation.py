"""Tests for permutation test utilities (M17)."""

from __future__ import annotations

import numpy as np
import pytest

from modelroot.stats.permutation import permutation_test_two_sample


class TestPermutationTwoSample:
    """Tests for two-sample permutation testing."""

    def test_reproducibility_with_seed(self) -> None:
        g1 = [1.0, 2.0, 3.0, 4.0, 5.0]
        g2 = [2.0, 3.0, 4.0, 5.0, 6.0]

        res1 = permutation_test_two_sample(g1, g2, n_permutations=500, seed=42)
        res2 = permutation_test_two_sample(g1, g2, n_permutations=500, seed=42)

        assert res1.observed_statistic == res2.observed_statistic
        assert res1.p_value == res2.p_value
        assert res1.n_permutations == res2.n_permutations == 500
        assert not res1.is_exact

    def test_monte_carlo_add_one_correction(self) -> None:
        # Heavily separated groups with large enough combinatorial space
        g1 = [100.0 + i for i in range(20)]
        g2 = [1.0 + i for i in range(20)]

        B = 99
        res = permutation_test_two_sample(g1, g2, alternative="greater", n_permutations=B, seed=123)
        # In 99 draws out of 1.37e11 combinations, 0 permutations will equal or exceed obs diff
        # So p_value must be exactly (0 + 1) / (99 + 1) = 0.01
        assert res.p_value == pytest.approx(1.0 / (B + 1), abs=1e-12)
        assert res.p_value > 0.0
        assert not res.is_exact

    def test_inclusive_tail_and_ties(self) -> None:
        # Identical groups with identical values
        g1 = [1.0, 1.0, 1.0]
        g2 = [1.0, 1.0, 1.0]

        B = 100
        res = permutation_test_two_sample(
            g1, g2, alternative="two-sided", n_permutations=B, seed=99
        )
        # All permutations will have diff = 0.0, identical to observed diff = 0.0.
        # Because ties are inclusive (>= |obs|), all 100 permutations are extreme:
        # p = (100 + 1) / (100 + 1) = 1.0
        assert res.p_value == pytest.approx(1.0, abs=1e-12)

    def test_alternative_directions(self) -> None:
        g1 = [10.0, 11.0, 12.0]
        g2 = [1.0, 2.0, 3.0]

        res_greater = permutation_test_two_sample(
            g1, g2, alternative="greater", n_permutations=200, seed=42
        )
        res_less = permutation_test_two_sample(
            g1, g2, alternative="less", n_permutations=200, seed=42
        )

        # g1 is clearly greater than g2, so greater should have small p, less should have large p
        assert res_greater.p_value < 0.1
        assert res_less.p_value > 0.9

    def test_exact_enumeration(self) -> None:
        # Small sample size: n1=3, n2=3 -> comb(6, 3) = 20 permutations
        g1 = [1.0, 2.0, 3.0]
        g2 = [4.0, 5.0, 6.0]

        res_exact = permutation_test_two_sample(
            g1, g2, alternative="two-sided", exact=True, retain_distribution=True
        )
        assert res_exact.is_exact is True
        assert res_exact.n_permutations == 20
        assert res_exact.null_distribution is not None
        assert len(res_exact.null_distribution) == 20

        # Exact p-value must be in [0, 1]
        assert 0.0 <= res_exact.p_value <= 1.0

    def test_exact_budget_limit_raises(self) -> None:
        g1 = list(range(20))
        g2 = list(range(20))
        # comb(40, 20) is ~1.37e11, far exceeding 10000
        with pytest.raises(ValueError, match="exceed max_exact_combinations"):
            permutation_test_two_sample(g1, g2, exact=True, max_exact_combinations=1000)

    def test_custom_statistic_function(self) -> None:
        g1 = [1.0, 2.0, 3.0]
        g2 = [10.0, 20.0, 30.0]

        # Custom statistic: difference in medians
        def diff_medians(a: np.ndarray, b: np.ndarray) -> float:
            return float(np.median(a) - np.median(b))

        res = permutation_test_two_sample(
            g1, g2, statistic_fn=diff_medians, n_permutations=100, seed=7
        )
        assert res.observed_statistic == pytest.approx(2.0 - 20.0, abs=1e-12)

    def test_input_validation(self) -> None:
        with pytest.raises(ValueError, match="at least one observation"):
            permutation_test_two_sample([], [1.0])

        with pytest.raises(ValueError, match="at least one observation"):
            permutation_test_two_sample([1.0], [])

        with pytest.raises(ValueError, match="NaN"):
            permutation_test_two_sample([1.0, float("nan")], [2.0])

        with pytest.raises(ValueError, match="Invalid alternative"):
            permutation_test_two_sample([1.0], [2.0], alternative="invalid")  # type: ignore[arg-type]

        with pytest.raises(ValueError, match="n_permutations must be >= 1"):
            permutation_test_two_sample([1.0], [2.0], n_permutations=0)
