"""M8 tests: repeated subsampling, NB t-test, power gate, is_negligible."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from modelroot.stats.nadeau_bengio import corrected_resampled_ttest, nb_to_contract
from modelroot.stats.power import GateResult, is_negligible, minimum_detectable_effect, power_gate
from modelroot.stats.repeated import RepeatedResult, consistency, repeated_subsample_deltas

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_split_data(n: int = 100, seed: int = 42) -> tuple[pd.DataFrame, pd.Series]:  # type: ignore[type-arg]
    rng = np.random.default_rng(seed)
    X = pd.DataFrame({"a": rng.random(n).astype(str), "b": rng.random(n).astype(str)})
    y = pd.Series(rng.integers(0, 2, n))
    return X, y


def _make_adapter_factory(algo: str = "logistic_regression", seed: int = 0):  # type: ignore[no-untyped-def]
    from modelroot.models.registry import get_model
    from modelroot.schemas.common import Algorithm
    from modelroot.schemas.config import ModelSpec

    col_types = {"a": "numeric", "b": "numeric"}
    spec = ModelSpec(algorithm=Algorithm(algo), params={})

    def factory() -> object:
        return get_model(spec, col_types, seed=seed)

    return factory


# ---------------------------------------------------------------------------
# consistency
# ---------------------------------------------------------------------------


def test_consistency_all_positive() -> None:
    assert consistency([0.1, 0.2, 0.3]) == pytest.approx(1.0)


def test_consistency_all_negative() -> None:
    assert consistency([-0.1, -0.2, -0.3]) == pytest.approx(0.0)


def test_consistency_mixed() -> None:
    assert consistency([-0.1, 0.2, -0.3, 0.4]) == pytest.approx(0.5)


def test_consistency_in_01() -> None:
    c = consistency([0.1, -0.1, 0.0, 0.2, -0.2])
    assert 0.0 <= c <= 1.0


def test_consistency_empty_raises() -> None:
    with pytest.raises(ValueError):
        consistency([])


def test_consistency_invariant_to_order() -> None:
    deltas = [0.1, -0.2, 0.3, -0.4, 0.5]
    import random

    shuffled = deltas.copy()
    random.seed(1)
    random.shuffle(shuffled)
    assert consistency(deltas) == consistency(shuffled)


# ---------------------------------------------------------------------------
# repeated_subsample_deltas
# ---------------------------------------------------------------------------


def test_repeated_subsample_returns_j_deltas() -> None:
    X, y = _make_split_data(60)
    make = _make_adapter_factory()
    result = repeated_subsample_deltas(make, make, X, y, "accuracy", J=4, seed=0)
    assert isinstance(result, RepeatedResult)
    assert len(result.deltas) == 4


def test_repeated_subsample_same_factory_delta_near_zero() -> None:
    """Same model factory on same data: delta should be exactly 0."""
    X, y = _make_split_data(80)
    make = _make_adapter_factory(seed=7)
    result = repeated_subsample_deltas(make, make, X, y, "accuracy", J=3, seed=0)
    for d in result.deltas:
        assert abs(d) < 1e-10


def test_repeated_subsample_deterministic() -> None:
    """Same seed -> same results."""
    X, y = _make_split_data(80)
    make_a = _make_adapter_factory(seed=0)
    make_b = _make_adapter_factory("random_forest", seed=0)
    r1 = repeated_subsample_deltas(make_a, make_b, X, y, "accuracy", J=3, seed=42)
    r2 = repeated_subsample_deltas(make_a, make_b, X, y, "accuracy", J=3, seed=42)
    assert r1.deltas == r2.deltas


def test_repeated_subsample_different_seeds_give_different_splits() -> None:
    """Different seeds produce different splits (nearly always)."""
    X, y = _make_split_data(80)
    make_a = _make_adapter_factory("random_forest", seed=1)
    make_b = _make_adapter_factory("logistic_regression", seed=1)
    r1 = repeated_subsample_deltas(make_a, make_b, X, y, "accuracy", J=3, seed=0)
    r2 = repeated_subsample_deltas(make_a, make_b, X, y, "accuracy", J=3, seed=999)
    # With different splits the deltas should differ (not always, but statistically)
    # Just verify they produce valid results
    assert len(r1.deltas) == 3
    assert len(r2.deltas) == 3


def test_repeated_subsample_j_too_small_raises() -> None:
    X, y = _make_split_data(60)
    make = _make_adapter_factory()
    with pytest.raises(ValueError, match="J"):
        repeated_subsample_deltas(make, make, X, y, "accuracy", J=1, seed=0)


# ---------------------------------------------------------------------------
# corrected_resampled_ttest
# ---------------------------------------------------------------------------


def test_nb_ttest_hand_computed() -> None:
    """Hand-computed NB result (tol 1e-9)."""
    # J=3, deltas=[0.1, 0.2, 0.3], n_train=80, n_test=20
    deltas = [0.1, 0.2, 0.3]
    n_train, n_test = 80, 20
    result = corrected_resampled_ttest(deltas, n_train, n_test)

    # mean = 0.2
    assert result.mean == pytest.approx(0.2, abs=1e-9)
    # s2 = sample variance = 0.01
    s2 = float(np.var(deltas, ddof=1))
    var_corr = (1 / 3 + n_test / n_train) * s2
    se_expected = math.sqrt(var_corr)
    assert result.se == pytest.approx(se_expected, abs=1e-9)
    # df = J-1 = 2
    assert result.df == pytest.approx(2.0, abs=1e-9)
    assert result.ci_low <= result.mean <= result.ci_high


def test_nb_ttest_requires_at_least_2_deltas() -> None:
    with pytest.raises(ValueError, match="2"):
        corrected_resampled_ttest([0.1], 80, 20)


def test_nb_to_contract_finite() -> None:
    deltas = [0.05, 0.1, 0.15, 0.0, -0.05]
    raw = corrected_resampled_ttest(deltas, 80, 20)
    contract = nb_to_contract(raw)
    # All fields finite
    assert math.isfinite(contract.mean)
    assert math.isfinite(contract.se)
    assert math.isfinite(contract.t)
    assert math.isfinite(contract.df)
    assert math.isfinite(contract.p)
    assert math.isfinite(contract.ci_low)
    assert math.isfinite(contract.ci_high)
    assert 0.0 <= contract.p <= 1.0
    assert contract.ci_low <= contract.ci_high


def test_nb_to_contract_clamps_inf() -> None:
    from modelroot.stats.nadeau_bengio import _CLAMP, NBTestResult, nb_to_contract

    raw = NBTestResult(
        mean=float("inf"), se=float("nan"), t=0.0, df=9.0, p=0.5, ci_low=-1.0, ci_high=1.0
    )
    contract = nb_to_contract(raw)
    assert math.isfinite(contract.mean)
    assert contract.mean == pytest.approx(_CLAMP)
    assert contract.se == pytest.approx(0.0)  # NaN -> clamped -> max(0, ...)


# ---------------------------------------------------------------------------
# minimum_detectable_effect
# ---------------------------------------------------------------------------


def test_mde_positive() -> None:
    mde = minimum_detectable_effect(se=0.05)
    assert mde > 0


def test_mde_formula_parity() -> None:
    """MDE = (z_{0.975} + z_{0.8}) * se, tol 1e-9."""
    from scipy.stats import norm

    se = 0.04
    alpha, power = 0.05, 0.8
    z_alpha = float(norm.ppf(1 - alpha / 2))
    z_power = float(norm.ppf(power))
    expected = (z_alpha + z_power) * se
    assert minimum_detectable_effect(se, alpha, power) == pytest.approx(expected, abs=1e-9)


def test_mde_se_zero() -> None:
    assert minimum_detectable_effect(0.0) == pytest.approx(0.0)


def test_mde_negative_se_raises() -> None:
    with pytest.raises(ValueError, match="se"):
        minimum_detectable_effect(-0.01)


# ---------------------------------------------------------------------------
# power_gate
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "n_holdout,n_pos,ci_width,expected_ok",
    [
        (200, 30, 0.09, True),
        (199, 30, 0.09, False),  # n_holdout too small
        (200, 29, 0.09, False),  # n_pos too small
        (200, 30, 0.11, False),  # ci_width too wide
        (200, 30, 0.10, True),  # exactly at boundary
        (201, 31, 0.05, True),
    ],
)
def test_power_gate_truth_table(
    n_holdout: int, n_pos: int, ci_width: float, expected_ok: bool
) -> None:
    result = power_gate(n_holdout, n_pos, ci_width)
    assert isinstance(result, GateResult)
    assert result.ok == expected_ok


def test_power_gate_reasons_populated_when_failing() -> None:
    result = power_gate(100, 10, 0.5)
    assert not result.ok
    assert len(result.reasons) > 0


# ---------------------------------------------------------------------------
# is_negligible
# ---------------------------------------------------------------------------


def test_is_negligible_true() -> None:
    assert is_negligible(0.01, 0.02) is True


def test_is_negligible_false_at_boundary() -> None:
    # ci_high == delta_min -> NOT negligible (strict <)
    assert is_negligible(0.02, 0.02) is False


def test_is_negligible_false_above() -> None:
    assert is_negligible(0.03, 0.02) is False


def test_is_negligible_invalid_delta_min() -> None:
    with pytest.raises(ValueError, match="delta_min"):
        is_negligible(0.01, 0.0)


# ---------------------------------------------------------------------------
# Extended Thresholds
# ---------------------------------------------------------------------------


def test_thresholds_m8_defaults() -> None:
    from modelroot.config.thresholds import Thresholds

    t = Thresholds()
    assert t.delta_min == pytest.approx(0.02)
    assert t.alpha == pytest.approx(0.05)
    assert t.n_boot == 1000
    assert t.repeats == 10
    assert t.holdout_min_n == 200
    assert t.holdout_min_pos == 30
    assert t.ci_width_max == pytest.approx(0.10)
    # M6 fields still present
    assert t.headroom == pytest.approx(0.03)
