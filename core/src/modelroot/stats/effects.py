"""Effect classification (M7).

``classify_effect`` maps a bootstrap result to one of four effect classes
based on the CI and practical-significance threshold ``delta_min``.
"""

from __future__ import annotations

from enum import Enum


class EffectClass(str, Enum):
    """Classification of a measured effect.

    Values follow the master plan §4.2:
      - ``practically_significant``: delta >= delta_min and CI lower > 0.
      - ``significant_but_small``: CI does not include 0 but delta < delta_min.
      - ``not_significant``: CI includes 0; effect size unclear.
      - ``negligible``: CI upper < delta_min (equivalence-style rejection).
    """

    practically_significant = "practically_significant"
    significant_but_small = "significant_but_small"
    not_significant = "not_significant"
    negligible = "negligible"


def classify_effect(
    delta: float,
    ci_low: float,
    ci_high: float,
    delta_min: float,
) -> EffectClass:
    """Classify the effect described by a bootstrap CI.

    Boundaries (applied in strict order, no epsilon):

    1. If ``ci_high < delta_min`` → ``negligible``
       (entire CI is below the practical threshold).
    2. If ``ci_low > 0`` and ``delta >= delta_min`` → ``practically_significant``
       (statistically significant and practically meaningful).
    3. If ``ci_low > 0`` and ``delta < delta_min`` → ``significant_but_small``
       (statistically significant but below the practical threshold).
    4. Otherwise → ``not_significant`` (CI includes 0).

    Args:
        delta: Point estimate of the effect (metric_b - metric_a).
        ci_low: Lower bound of the bootstrap CI.
        ci_high: Upper bound of the bootstrap CI.
        delta_min: Practical-significance threshold (must be > 0).

    Returns:
        One of the four :class:`EffectClass` values.

    Raises:
        ValueError: If ``delta_min <= 0`` or ``ci_low > ci_high``.
    """
    if delta_min <= 0:
        raise ValueError(f"delta_min must be > 0, got {delta_min}")
    if ci_low > ci_high:
        raise ValueError(f"ci_low ({ci_low}) must be <= ci_high ({ci_high})")

    if ci_high < delta_min:
        return EffectClass.negligible
    if ci_low > 0 and delta >= delta_min:
        return EffectClass.practically_significant
    if ci_low > 0:
        return EffectClass.significant_but_small
    return EffectClass.not_significant
