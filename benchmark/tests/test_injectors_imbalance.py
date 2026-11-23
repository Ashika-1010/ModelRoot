"""Tests for H2 imbalance and separable hard-negative injectors (M14)."""

from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import get_dataset, list_datasets
from modelroot_bench.inject import get_injector, list_injectors
from modelroot_bench.inject.hard_negative_separable import HardNegativeSeparableInjector
from modelroot_bench.inject.imbalance import ImbalanceInjector
from modelroot_bench.oracle import impact_from_results, oracle_impact


def _hash_df(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame for determinism testing."""
    return hashlib.sha256(pd.util.hash_pandas_object(df, index=True).values).hexdigest()


def test_injector_registration() -> None:
    """Verify imbalance injectors are registered."""
    injectors = list_injectors()
    assert "imbalance" in injectors
    assert "hard_negative_separable" in injectors
    assert isinstance(get_injector("imbalance"), ImbalanceInjector)
    assert isinstance(get_injector("hard_negative_separable"), HardNegativeSeparableInjector)


def test_synth_large_dataset() -> None:
    """Verify synth_large dataset is registered and has 3000 rows."""
    assert "synth_large" in list_datasets()
    ds = get_dataset("synth_large")
    df = ds.load()
    assert len(df) == 3000
    assert len([c for c in df.columns if c != "target"]) == 15
    assert set(df["target"].unique()).issubset({0, 1})


def test_imbalance_injector_severities() -> None:
    """Verify S1, S2, S3 imbalance ratios and structure."""
    ds = get_dataset("synthetic_medium")
    df = ds.load()
    inj = ImbalanceInjector()

    for sev in ("S1", "S2", "S3"):
        scen = inj.apply(df, ds, severity=sev, seed=42)

        # Ground truth check
        assert len(scen.ground_truth.injected) == 1
        assert scen.ground_truth.injected[0].label == "imbalance"
        assert scen.ground_truth.injected[0].severity == sev
        if sev == "S1":
            assert scen.ground_truth.expected_subtypes == {
                "imbalance:distortion": True,
                "imbalance:harm": True,
            }
        elif sev in ("S2", "S3"):
            assert scen.ground_truth.expected_subtypes == {"imbalance:distortion": True}

        # Split column present
        assert "split" in scen.df.columns
        assert set(scen.df["split"].unique()) == {"train", "test"}

        # Check minority share in train
        train_df = scen.df[scen.df["split"] == "train"]
        counts = train_df["target"].value_counts()
        min_share = counts.min() / counts.sum()

        if sev == "S1":
            assert 0.14 <= min_share <= 0.20  # ~1:5 ratio (16.7%)
        elif sev == "S2":
            assert 0.03 <= min_share <= 0.07  # ~1:20 ratio (4.8%)
        elif sev == "S3":
            assert min_share <= 0.02  # ~1:100 ratio (1.0%)

        # Clean df is uncorrupted
        assert len(scen.clean_df) == len(df)


def test_imbalance_injector_determinism() -> None:
    """Verify applying injector with same seed produces identical data."""
    ds = get_dataset("synthetic_medium")
    df = ds.load()
    inj = ImbalanceInjector()

    scen1 = inj.apply(df, ds, severity="S1", seed=123)
    scen2 = inj.apply(df, ds, severity="S1", seed=123)

    assert _hash_df(scen1.df) == _hash_df(scen2.df)
    assert scen1.ground_truth == scen2.ground_truth


def test_separable_hard_negative_injector() -> None:
    """Verify separable hard negative injector generates valid negative scenario."""
    ds = get_dataset("synthetic_medium")
    df = ds.load()
    inj = HardNegativeSeparableInjector()

    scen = inj.apply(df, ds, severity="", seed=42)

    # Injected failures must be empty for negative scenario
    assert scen.ground_truth.injected == []
    assert "separable_imbalance" in scen.ground_truth.distractors
    assert scen.ground_truth.expected_subtypes == {
        "imbalance:distortion": False,
        "imbalance:harm": False,
    }

    # Target class distribution should be imbalanced (~1:10)
    counts = scen.df["target"].value_counts()
    min_share = counts.min() / counts.sum()
    assert 0.05 <= min_share <= 0.15

    # Oracle impact should be minimal due to class_sep=3.0
    impact, no_effect = oracle_impact(
        clean_df=scen.clean_df,
        scenario_df=scen.df,
        target=ds.target,
        scenario_model_spec=scen.model_spec,
        seed=42,
    )
    assert impact < 0.05
    assert no_effect or impact < 0.02


def test_direction_aware_impact_from_results() -> None:
    """Verify direction-aware impact calculation for higher and lower-is-better metrics."""
    # Higher-is-better: clean > scenario means degradation
    imp1, no_eff1 = impact_from_results(clean_score=0.85, scenario_score=0.75, metric="macro_f1")
    assert np.isclose(imp1, 0.10)
    assert not no_eff1

    # Higher-is-better: minimal degradation -> no_effect=True
    imp2, no_eff2 = impact_from_results(clean_score=0.85, scenario_score=0.84, metric="macro_f1")
    assert np.isclose(imp2, 0.01)
    assert no_eff2

    # Lower-is-better: scenario > clean means degradation
    imp3, no_eff3 = impact_from_results(clean_score=0.20, scenario_score=0.35, metric="log_loss")
    assert np.isclose(imp3, 0.15)
    assert not no_eff3

    # Lower-is-better: scenario improved (smaller) -> negative impact -> no_effect=True
    imp4, no_eff4 = impact_from_results(clean_score=0.30, scenario_score=0.25, metric="brier_score")
    assert np.isclose(imp4, -0.05)
    assert no_eff4
