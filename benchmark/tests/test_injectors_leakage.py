"""Tests for M15 benchmark injectors (leak_duplicate, leak_identifier, hard_negative_highcard)."""

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.hard_negative_highcard import HardNegativeHighcardInjector
from modelroot_bench.inject.leak_duplicate import LeakDuplicateInjector
from modelroot_bench.inject.leak_identifier import LeakIdentifierInjector


def _make_dummy_spec() -> DatasetSpec:
    return DatasetSpec(
        name="test_data",
        load=lambda: pd.DataFrame({"x": [1, 2], "target": [0, 1]}),
        target="target",
        positive="1",
        role="dev",
        description="test",
    )


def test_leak_duplicate_injector() -> None:
    """Validate duplicate injection, holdout inflation, expected subtypes."""
    n = 3000
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )

    inj = LeakDuplicateInjector()
    spec = _make_dummy_spec()

    scen = inj.apply(df, spec, severity="S2", seed=42)
    assert scen.direction == "inflation"
    assert scen.ground_truth.expected_subtypes.get("leakage:duplicate") is True
    assert len(scen.ground_truth.injected) == 1
    assert scen.ground_truth.injected[0].label == "leakage:duplicate"

    # Check that holdout split contains exact rows from train split
    df_res = scen.df
    df_tr = df_res[df_res["split"] == "train"].drop(columns=["split"])
    df_ho = df_res[df_res["split"] == "test"].drop(columns=["split"])

    # At least some rows in df_ho are identical to df_tr
    tr_set = set(zip(df_tr["f1"].round(6), df_tr["f2"].round(6), df_tr["target"], strict=True))
    ho_dups = [
        row
        for row in zip(df_ho["f1"].round(6), df_ho["f2"].round(6), df_ho["target"], strict=True)
        if row in tr_set
    ]
    assert len(ho_dups) > 50


def test_leak_identifier_injector() -> None:
    """Validate row-order leakage and row_id feature addition."""
    n = 1500
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )

    inj = LeakIdentifierInjector()
    spec = _make_dummy_spec()

    scen = inj.apply(df, spec, severity="S3", seed=42)
    assert scen.direction == "inflation"
    assert "row_id" in scen.df.columns
    assert scen.ground_truth.expected_subtypes.get("leakage:identifier") is True
    assert scen.ground_truth.injected[0].label == "leakage:identifier"


def test_hard_negative_highcard_injector() -> None:
    """Validate token feature addition with no injected failures."""
    n = 500
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )

    inj = HardNegativeHighcardInjector()
    spec = _make_dummy_spec()

    scen = inj.apply(df, spec, severity="", seed=42)
    assert "token" in scen.df.columns
    assert scen.df["token"].nunique() == n
    assert len(scen.ground_truth.injected) == 0
    assert scen.ground_truth.expected_subtypes.get("leakage:duplicate") is False
    assert scen.ground_truth.expected_subtypes.get("leakage:identifier") is False
