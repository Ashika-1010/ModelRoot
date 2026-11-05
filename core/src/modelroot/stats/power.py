"""Power, MDE and power-gate calculations (M8).

``minimum_detectable_effect`` computes the MDE given a standard error and
desired power.  ``power_gate`` checks whether the experimental setup is
adequately powered before committing to a significance decision.
``is_negligible`` provides the equivalence-style rejection check.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from scipy.stats import norm


@dataclass(frozen=True)
class GateResult:
    """Output of the power gate check.

    Attributes:
        ok: ``True`` if all power conditions are met.
        reasons: Human-readable reasons for any failures.
    """

    ok: bool
    reasons: list[str] = field(default_factory=list)


def minimum_detectable_effect(
    se: float,
    alpha: float = 0.05,
    power: float = 0.8,
) -> float:
    """Compute the minimum detectable effect for given SE and desired power.

    MDE = (z_{1-alpha/2} + z_{power}) * se

    Args:
        se: Estimated standard error of the effect (>= 0).
        alpha: Significance level (0 < alpha < 1).
        power: Desired statistical power (0 < power < 1).

    Returns:
        MDE as a positive float.

    Raises:
        ValueError: If se < 0, or alpha / power out of range.
    """
    if se < 0:
        raise ValueError(f"se must be >= 0, got {se}")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    if not (0.0 < power < 1.0):
        raise ValueError(f"power must be in (0, 1), got {power}")

    z_alpha = float(norm.ppf(1.0 - alpha / 2.0))
    z_power = float(norm.ppf(power))
    return (z_alpha + z_power) * se


def power_gate(
    n_holdout: int,
    n_pos: int,
    ci_width: float,
    holdout_min_n: int = 200,
    holdout_min_pos: int = 30,
    ci_width_max: float = 0.10,
) -> GateResult:
    """Check whether the evaluation setup is adequately powered.

    Three independent conditions must all pass:
      1. ``n_holdout >= holdout_min_n``
      2. ``n_pos >= holdout_min_pos``
      3. ``ci_width <= ci_width_max``

    Args:
        n_holdout: Total holdout set size.
        n_pos: Number of positive-class samples in the holdout set.
        ci_width: Bootstrap CI width (ci_high - ci_low).
        holdout_min_n: Minimum total holdout samples.
        holdout_min_pos: Minimum positive-class holdout samples.
        ci_width_max: Maximum acceptable CI width.

    Returns:
        :class:`GateResult` with ``ok`` flag and reason strings.

    Raises:
        ValueError: If any threshold is non-positive or ci_width is negative.
    """
    if holdout_min_n <= 0:
        raise ValueError(f"holdout_min_n must be > 0, got {holdout_min_n}")
    if holdout_min_pos <= 0:
        raise ValueError(f"holdout_min_pos must be > 0, got {holdout_min_pos}")
    if ci_width_max <= 0:
        raise ValueError(f"ci_width_max must be > 0, got {ci_width_max}")
    if ci_width < 0:
        raise ValueError(f"ci_width must be >= 0, got {ci_width}")

    reasons: list[str] = []

    if n_holdout < holdout_min_n:
        reasons.append(f"Holdout too small: {n_holdout} < {holdout_min_n} (min_n).")
    if n_pos < holdout_min_pos:
        reasons.append(f"Too few positive holdout samples: {n_pos} < {holdout_min_pos} (min_pos).")
    if ci_width > ci_width_max:
        reasons.append(f"CI too wide: {ci_width:.4f} > {ci_width_max} (max_width).")

    return GateResult(ok=len(reasons) == 0, reasons=reasons)


def is_negligible(ci_high: float, delta_min: float) -> bool:
    """Return True if the CI upper bound is strictly below the practical threshold.

    This is the equivalence-style rejection check: if the entire CI lies
    below ``delta_min``, the effect is negligible.

    Args:
        ci_high: Upper bound of the bootstrap CI.
        delta_min: Practical-significance threshold (> 0).

    Returns:
        ``True`` if ``ci_high < delta_min``.

    Raises:
        ValueError: If ``delta_min <= 0``.
    """
    if delta_min <= 0:
        raise ValueError(f"delta_min must be > 0, got {delta_min}")
    return ci_high < delta_min
