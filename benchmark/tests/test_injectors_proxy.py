"""Tests for M16 benchmark injectors (leak_proxy, hard_negative_strong)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.hard_negative_strong import HardNegativeStrongInjector
from modelroot_bench.inject.leak_proxy import LeakProxyInjector


def _make_dummy_spec() -> DatasetSpec:
    return DatasetSpec(
        name="test_data",
        load=lambda: pd.DataFrame({"x": [1, 2], "target": [0, 1]}),
        target="target",
        positive="1",
        role="dev",
        description="test",
    )


def _make_df(n: int = 500, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "f2": rng.standard_normal(n),
            "target": rng.integers(0, 2, size=n),
        }
    )


# ---------------------------------------------------------------------------
# LeakProxyInjector
# ---------------------------------------------------------------------------


def test_leak_proxy_s1_adds_noisy_proxy() -> None:
    """S1: proxy column added with high noise."""
    df = _make_df()
    spec = _make_dummy_spec()
    inj = LeakProxyInjector()
    scen = inj.apply(df, spec, severity="S1", seed=0)

    assert scen.direction == "inflation"
    assert "proxy" in scen.df.columns
    assert scen.ground_truth.injected[0].label == "leakage:target_proxy"
    assert scen.ground_truth.expected_subtypes.get("target_proxy") is True
    assert scen.current_df is None


def test_leak_proxy_s3_perfect_proxy() -> None:
    """S3: proxy = y exactly (sigma=0)."""
    df = _make_df()
    spec = _make_dummy_spec()
    inj = LeakProxyInjector()
    scen = inj.apply(df, spec, severity="S3", seed=0)

    df_tr = scen.df[scen.df["split"] == "train"]
    assert "proxy" in df_tr.columns
    # Perfect proxy: proxy == target exactly (within floating point)
    target_match = (df_tr["proxy"].round(6) == df_tr["target"].astype(float).round(6)).all()
    assert target_match, "S3 proxy should exactly match target labels"


def test_leak_proxy_s3f_flagged_in_metadata() -> None:
    """S3f: flagged variant sets unavailable_at_inference metadata."""
    df = _make_df()
    spec = _make_dummy_spec()
    inj = LeakProxyInjector()
    scen = inj.apply(df, spec, severity="S3f", seed=0)

    assert scen.metadata.get("unavailable_at_inference") == ["proxy"]
    assert "proxy" in scen.df.columns
    assert scen.ground_truth.injected[0].label == "leakage:target_proxy"


def test_leak_proxy_s3t_has_current_df() -> None:
    """S3t: transfer variant produces a current_df with high-noise proxy."""
    df = _make_df(n=3000)
    spec = _make_dummy_spec()
    inj = LeakProxyInjector()
    scen = inj.apply(df, spec, severity="S3t", seed=0)

    assert scen.current_df is not None, "Transfer variant should produce current_df"
    assert "proxy" in scen.current_df.columns
    # In current_df, proxy should have higher noise — correlates less with target
    y_curr = scen.current_df["target"].to_numpy(dtype=float)
    proxy_curr = scen.current_df["proxy"].to_numpy(dtype=float)
    corr = float(np.corrcoef(y_curr, proxy_curr)[0, 1])
    # High-noise sigma=2 on binary target → low correlation
    assert corr < 0.5, f"Transfer current proxy should be noisy; got corr={corr:.3f}"


def test_leak_proxy_deterministic() -> None:
    """Same seed produces identical scenario."""
    df = _make_df()
    spec = _make_dummy_spec()
    inj = LeakProxyInjector()
    s1 = inj.apply(df, spec, severity="S2", seed=42)
    s2 = inj.apply(df, spec, severity="S2", seed=42)
    pd.testing.assert_frame_equal(s1.df, s2.df)


# ---------------------------------------------------------------------------
# HardNegativeStrongInjector
# ---------------------------------------------------------------------------


def test_hard_negative_strong_no_failures() -> None:
    """Hard-negative: no injected failures, strong_pred column present."""
    df = _make_df()
    spec = _make_dummy_spec()
    inj = HardNegativeStrongInjector()
    scen = inj.apply(df, spec, severity="S1", seed=0)

    assert len(scen.ground_truth.injected) == 0
    assert scen.ground_truth.no_effect is True
    assert "strong_pred" in scen.df.columns
    assert scen.direction == "degradation"
    assert scen.metadata == {}


def test_hard_negative_strong_predictor_correlated_with_target() -> None:
    """strong_pred should be correlated with target (small noise)."""
    df = _make_df(n=500)
    spec = _make_dummy_spec()
    inj = HardNegativeStrongInjector()
    scen = inj.apply(df, spec, severity="", seed=0)

    df_tr = scen.df[scen.df["split"] == "train"]
    y = df_tr["target"].to_numpy(dtype=float)
    sp = df_tr["strong_pred"].to_numpy(dtype=float)
    corr = float(np.corrcoef(y, sp)[0, 1])
    assert corr > 0.70, f"strong_pred should correlate with target; got {corr:.3f}"


def test_hard_negative_strong_deterministic() -> None:
    """Same seed produces identical scenario."""
    df = _make_df()
    spec = _make_dummy_spec()
    inj = HardNegativeStrongInjector()
    s1 = inj.apply(df, spec, severity="S1", seed=99)
    s2 = inj.apply(df, spec, severity="S1", seed=99)
    pd.testing.assert_frame_equal(s1.df, s2.df)
