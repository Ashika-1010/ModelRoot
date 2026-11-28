"""Permutation testing utilities for two-sample and paired hypothesis tests.

Provides reproducible permutation tests with inclusive-tail handling, ties support,
Monte Carlo add-one correction for estimated p-values, and optional exact enumeration
for small combinatorial sample spaces.
"""

from __future__ import annotations

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Literal

import numpy as np


@dataclass(frozen=True)
class PermutationResult:
    """Result of a permutation test.

    Attributes:
        observed_statistic: The observed test statistic on the original data.
        p_value: The computed p-value (Monte Carlo with add-one correction or exact).
        n_permutations: Number of evaluated permutations.
        is_exact: True if all permutations were exhaustively enumerated, False for Monte Carlo.
        alternative: Direction of the alternative hypothesis ("greater", "less", "two-sided").
        null_distribution: Array of permutation statistics (or None if not retained).
    """

    observed_statistic: float
    p_value: float
    n_permutations: int
    is_exact: bool
    alternative: str
    null_distribution: np.ndarray | None = None


def _diff_means(group1: np.ndarray, group2: np.ndarray) -> float:
    """Default two-sample statistic: mean(group1) - mean(group2)."""
    return float(np.mean(group1) - np.mean(group2))


def permutation_test_two_sample(
    group1: Sequence[float],
    group2: Sequence[float],
    *,
    statistic_fn: Callable[[np.ndarray, np.ndarray], float] | None = None,
    alternative: Literal["two-sided", "greater", "less"] = "two-sided",
    n_permutations: int = 1000,
    seed: int | np.random.Generator | None = None,
    exact: bool = False,
    max_exact_combinations: int = 10000,
    retain_distribution: bool = False,
) -> PermutationResult:
    """Perform a two-sample permutation test for difference between two groups.

    Args:
        group1: First sample observations.
        group2: Second sample observations.
        statistic_fn: Function (g1, g2) -> float. Defaults to difference in means.
        alternative: "two-sided", "greater" (g1 > g2), or "less" (g1 < g2).
        n_permutations: Number of Monte Carlo permutations if exact is False. Must be >= 1.
        seed: Random seed or Generator for reproducible Monte Carlo sampling.
        exact: If True, exhaustively enumerates all allocations when total combinations
            <= max_exact_combinations.
        max_exact_combinations: Maximum number of combinations allowed for exact enumeration.
        retain_distribution: Whether to include the null distribution in the result.

    Returns:
        PermutationResult containing observed statistic, p-value, and metadata.

    Raises:
        ValueError: On empty inputs, invalid alternative, or non-positive permutations.
    """
    g1 = np.asarray(group1, dtype=np.float64)
    g2 = np.asarray(group2, dtype=np.float64)

    if g1.ndim != 1 or g2.ndim != 1:
        raise ValueError("group1 and group2 must be 1-dimensional sequences")

    n1 = len(g1)
    n2 = len(g2)
    if n1 == 0 or n2 == 0:
        raise ValueError("Both group1 and group2 must contain at least one observation")

    if np.any(np.isnan(g1)) or np.any(np.isnan(g2)):
        raise ValueError("Inputs must not contain NaN")

    if alternative not in ("two-sided", "greater", "less"):
        raise ValueError(
            f"Invalid alternative '{alternative}'. Must be 'two-sided', 'greater', or 'less'"
        )

    stat_fn = statistic_fn if statistic_fn is not None else _diff_means
    obs_stat = float(stat_fn(g1, g2))

    pooled = np.concatenate([g1, g2])
    n_total = n1 + n2

    total_combinations = math.comb(n_total, n1)

    if exact:
        if total_combinations > max_exact_combinations:
            raise ValueError(
                f"Exact enumeration requested, but total combinations ({total_combinations}) "
                f"exceed max_exact_combinations ({max_exact_combinations}). "
                "Use Monte Carlo permutation (exact=False) instead."
            )

        # Exhaustive enumeration of all combinations of size n1
        null_stats: list[float] = []
        indices = range(n_total)
        for combo in itertools.combinations(indices, n1):
            mask = np.zeros(n_total, dtype=bool)
            mask[list(combo)] = True
            sub1 = pooled[mask]
            sub2 = pooled[~mask]
            null_stats.append(float(stat_fn(sub1, sub2)))

        null_arr = np.asarray(null_stats, dtype=np.float64)
        n_eval = len(null_arr)

        # Exact p-value: proportion of exact permutations at least as extreme (with ties)
        tol = 1e-12
        if alternative == "greater":
            extreme_count = np.sum(null_arr >= obs_stat - tol)
        elif alternative == "less":
            extreme_count = np.sum(null_arr <= obs_stat + tol)
        else:  # two-sided
            extreme_count = np.sum(np.abs(null_arr) >= np.abs(obs_stat) - tol)

        p_value = float(extreme_count) / float(n_eval)
        p_value = min(1.0, max(0.0, p_value))

        return PermutationResult(
            observed_statistic=obs_stat,
            p_value=p_value,
            n_permutations=n_eval,
            is_exact=True,
            alternative=alternative,
            null_distribution=null_arr if retain_distribution else None,
        )

    # Monte Carlo Permutation Test
    if n_permutations < 1:
        raise ValueError(f"n_permutations must be >= 1, got {n_permutations}")

    rng = np.random.default_rng(seed)
    null_stats_mc = np.empty(n_permutations, dtype=np.float64)

    # Reshuffle pooled array
    perm_arr = pooled.copy()
    for b in range(n_permutations):
        rng.shuffle(perm_arr)
        null_stats_mc[b] = stat_fn(perm_arr[:n1], perm_arr[n1:])

    # Inclusive ties handling with Monte Carlo add-one correction:
    # p = (sum(I(extreme)) + 1) / (B + 1)
    tol = 1e-12
    extreme_mc: int
    if alternative == "greater":
        extreme_mc = int(np.sum(null_stats_mc >= obs_stat - tol))
    elif alternative == "less":
        extreme_mc = int(np.sum(null_stats_mc <= obs_stat + tol))
    else:  # two-sided
        extreme_mc = int(np.sum(np.abs(null_stats_mc) >= np.abs(obs_stat) - tol))

    p_value = float(extreme_mc + 1) / float(n_permutations + 1)
    p_value = min(1.0, max(0.0, p_value))

    return PermutationResult(
        observed_statistic=obs_stat,
        p_value=p_value,
        n_permutations=n_permutations,
        is_exact=False,
        alternative=alternative,
        null_distribution=null_stats_mc if retain_distribution else None,
    )
