"""Nadeau-Bengio corrected resampled t-test (M8).

Reference: Nadeau & Bengio (2003), "Inference for the Generalization Error".

The corrected variance accounts for the overlap between training sets in
repeated subsampling:

    var_corr = (1/J + n_test/n_train) * s^2

where s^2 is the sample variance of the J deltas, n_train / n_test are the
sizes of the training and test partitions.

The JSON-safe conversion ``nb_to_contract`` maps the result to the M1
``NBResult`` schema, clamping any non-finite values to ±1e15.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy.stats import t as t_dist

from modelroot.schemas.experiment import NBResult

# Finite clamp applied before constructing M1 contracts
_CLAMP = 1e15


@dataclass(frozen=True)
class NBTestResult:
    """Raw Nadeau-Bengio corrected t-test output (internal).

    Attributes:
        mean: Mean delta across J repeats.
        se: Corrected standard error sqrt(var_corr).
        t: t-statistic (mean / se).
        df: Degrees of freedom (J - 1).
        p: Two-sided p-value.
        ci_low: Lower bound of the (1-alpha) CI.
        ci_high: Upper bound of the (1-alpha) CI.
    """

    mean: float
    se: float
    t: float
    df: float
    p: float
    ci_low: float
    ci_high: float


def corrected_resampled_ttest(
    deltas: list[float],
    n_train: int,
    n_test: int,
    alpha: float = 0.05,
) -> NBTestResult:
    """Apply the Nadeau-Bengio correction to repeated-subsampling deltas.

    Args:
        deltas: List of J per-repeat metric deltas.
        n_train: Number of training rows per repeat.
        n_test: Number of test rows per repeat.
        alpha: Significance level for the CI (0 < alpha < 1).

    Returns:
        :class:`NBTestResult` with corrected statistics.

    Raises:
        ValueError: If len(deltas) < 2, n_train < 1, n_test < 1, or alpha
            is out of (0, 1).
    """
    J = len(deltas)
    if J < 2:
        raise ValueError(f"Need at least 2 deltas for the NB t-test; got {J}.")
    if n_train < 1:
        raise ValueError(f"n_train must be >= 1, got {n_train}.")
    if n_test < 1:
        raise ValueError(f"n_test must be >= 1, got {n_test}.")
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}.")

    arr = np.asarray(deltas, dtype=float)
    mean_delta = float(np.mean(arr))
    s2 = float(np.var(arr, ddof=1))  # sample variance

    var_corr = (1.0 / J + n_test / n_train) * s2
    se = math.sqrt(max(var_corr, 0.0))

    df = float(J - 1)

    if se <= 0.0:
        t_stat = 0.0
        p_val = 1.0
    else:
        t_stat = mean_delta / se
        p_val = float(2.0 * t_dist.sf(abs(t_stat), df=df))

    # CI using t-distribution
    t_crit = float(t_dist.ppf(1.0 - alpha / 2.0, df=df))
    ci_low = mean_delta - t_crit * se
    ci_high = mean_delta + t_crit * se

    return NBTestResult(
        mean=mean_delta,
        se=se,
        t=t_stat,
        df=df,
        p=p_val,
        ci_low=ci_low,
        ci_high=ci_high,
    )


def nb_to_contract(nb: NBTestResult) -> NBResult:
    """Convert an :class:`NBTestResult` to the M1 :class:`NBResult` contract.

    Non-finite values are clamped to ±_CLAMP to preserve JSON safety.
    ``ci_low`` is always <= ``ci_high`` after clamping.

    Args:
        nb: Raw NB test result.

    Returns:
        A validated :class:`NBResult` Pydantic model.
    """

    def _safe(v: float) -> float:
        if not math.isfinite(v):
            return _CLAMP if v > 0 else -_CLAMP
        return v

    ci_low = _safe(nb.ci_low)
    ci_high = _safe(nb.ci_high)
    # Enforce ordering after clamping
    if ci_low > ci_high:
        ci_low, ci_high = ci_high, ci_low

    return NBResult(
        mean=_safe(nb.mean),
        se=max(0.0, _safe(nb.se)),
        t=_safe(nb.t),
        df=_safe(nb.df),
        p=float(np.clip(_safe(nb.p), 0.0, 1.0)),
        ci_low=ci_low,
        ci_high=ci_high,
    )
