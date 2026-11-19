"""Tests for failure injectors (M13)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import get_dataset
from modelroot_bench.inject.clean import CleanControlInjector
from modelroot_bench.inject.overfitting import OverfittingInjector


def _hash_df(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame for determinism testing."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def test_clean_control_injector() -> None:
    ds = get_dataset("synthetic_easy")
    df = ds.load()
    inj = CleanControlInjector()
    scen = inj.apply(df, ds, severity="", seed=42)

    assert scen.ground_truth.injected == []
    assert scen.ground_truth.impact == 0.0
    assert not scen.ground_truth.no_effect
    assert len(scen.df) == len(df)


def test_overfitting_injector_severities_and_features() -> None:
    ds = get_dataset("synthetic_easy")
    df = ds.load()
    inj = OverfittingInjector()

    for sev in ("S1", "S2", "S3"):
        scen = inj.apply(df, ds, severity=sev, seed=42)

        # Ground truth check
        assert len(scen.ground_truth.injected) == 1
        assert scen.ground_truth.injected[0].label == "overfitting"
        assert scen.ground_truth.injected[0].severity == sev

        # Noise columns check (30 noise features)
        noise_cols = [c for c in scen.df.columns if c.startswith("noise_")]
        assert len(noise_cols) == 30

        # Split column check
        assert "split" in scen.df.columns
        split_vals = set(scen.df["split"].unique())
        assert split_vals.issubset({"train", "test"})

        n_train = int(np.sum(scen.df["split"] == "train"))
        if sev == "S1":
            assert n_train == 1000
        elif sev == "S2":
            assert n_train == 400
        elif sev == "S3":
            assert n_train == 150

        # High-capacity model spec
        assert scen.model_spec["algorithm"] == "random_forest"
        assert scen.model_spec["params"]["min_samples_leaf"] == 1


def test_overfitting_injector_determinism() -> None:
    """Same seed must produce bitwise identical scenario DataFrames."""
    ds = get_dataset("synthetic_easy")
    df = ds.load()
    inj = OverfittingInjector()

    scen1 = inj.apply(df, ds, severity="S2", seed=99)
    scen2 = inj.apply(df, ds, severity="S2", seed=99)

    h1 = _hash_df(scen1.df)
    h2 = _hash_df(scen2.df)
    assert h1 == h2
