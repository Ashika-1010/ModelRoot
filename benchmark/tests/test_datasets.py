"""Tests for benchmark dataset registry and built-in datasets (M13)."""

from __future__ import annotations

import pandas as pd
import pytest

from modelroot_bench.datasets.registry import (
    DatasetSpec,
    get_dataset,
    list_datasets,
    register_dataset,
)


def test_list_and_get_builtin_datasets() -> None:
    names = list_datasets()
    assert "synthetic_easy" in names
    assert "synthetic_medium" in names
    assert "synthetic_hard" in names
    assert "sklearn_breast_cancer" in names

    easy = get_dataset("synthetic_easy")
    assert easy.name == "synthetic_easy"
    assert easy.target == "target"
    assert easy.role == "dev"


def test_load_synthetic_easy() -> None:
    ds = get_dataset("synthetic_easy")
    df = ds.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1500
    assert "target" in df.columns
    assert set(df["target"].unique()).issubset({0, 1})


def test_load_breast_cancer() -> None:
    ds = get_dataset("sklearn_breast_cancer")
    df = ds.load()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 569
    assert ds.target in df.columns


def test_register_duplicate_raises() -> None:
    spec = DatasetSpec(name="dup_test", load=lambda: pd.DataFrame(), target="t")
    register_dataset(spec)
    with pytest.raises(ValueError, match="already registered"):
        register_dataset(spec)


def test_unknown_dataset_raises() -> None:
    with pytest.raises(KeyError, match="not registered"):
        get_dataset("nonexistent_dataset")
