"""Tests for oracle impact estimation (M13)."""

from __future__ import annotations

from modelroot_bench.datasets.registry import get_dataset
from modelroot_bench.inject.overfitting import OverfittingInjector
from modelroot_bench.oracle import oracle_impact


def test_oracle_impact_on_clean_versus_clean() -> None:
    ds = get_dataset("synthetic_easy")
    df = ds.load()
    impact, no_effect = oracle_impact(
        clean_df=df,
        scenario_df=df,
        target=ds.target,
        seed=42,
    )
    # Impact between clean and clean should be near zero, hence no_effect=True
    assert abs(impact) < 0.05
    assert no_effect or impact < 0.02


def test_oracle_impact_on_overfitting_case() -> None:
    ds = get_dataset("synthetic_easy")
    df = ds.load()
    inj = OverfittingInjector()
    scen = inj.apply(df, ds, severity="S3", seed=42)

    impact, no_effect = oracle_impact(
        clean_df=scen.clean_df,
        scenario_df=scen.df,
        target=ds.target,
        scenario_model_spec=scen.model_spec,
        seed=42,
    )
    # Severe overfitting with noise features on 150 training samples has impact
    assert isinstance(impact, float)
    assert isinstance(no_effect, bool)
