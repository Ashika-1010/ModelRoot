"""Shared fixtures for assessment tests (M5/M6)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from modelroot.data.loader import load_csv
from modelroot.data.split import SplitData, make_split
from modelroot.data.validate import validate_for_analysis
from modelroot.schemas.common import Algorithm, SplitMode
from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig
from modelroot.utils.seeds import SeedManager


def _make_csv(tmp_path: pathlib.Path, n: int = 60, seed: int = 42) -> pathlib.Path:
    """Write a balanced binary-classification CSV and return its path."""
    rng = np.random.default_rng(seed)
    ages = rng.integers(20, 70, size=n).astype(str)
    incomes = rng.integers(30000, 120000, size=n).astype(str)
    labels = (rng.random(n) > 0.5).astype(int)
    label_strs = ["yes" if lbl == 1 else "no" for lbl in labels]
    df = pd.DataFrame({"age": ages, "income": incomes, "label": label_strs})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


@pytest.fixture()
def csv_path(tmp_path: pathlib.Path) -> pathlib.Path:
    """Return path to a small balanced CSV fixture."""
    return _make_csv(tmp_path)


@pytest.fixture()
def analysis_config(csv_path: pathlib.Path) -> AnalysisConfig:
    """Return an AnalysisConfig for the small CSV fixture."""
    return AnalysisConfig(
        data_path=csv_path,
        target="label",
        positive_class="yes",
        split=SplitConfig(mode=SplitMode.single_file),
        model_spec=ModelSpec(algorithm=Algorithm.logistic_regression, params={}),
        seed=0,
    )


@pytest.fixture()
def split_data(csv_path: pathlib.Path, analysis_config: AnalysisConfig) -> SplitData:
    """Return a SplitData for the small CSV fixture."""
    df = load_csv(csv_path)
    validated = validate_for_analysis(df, analysis_config)
    sm = SeedManager(0)
    return make_split(validated, analysis_config, seed=sm.spawn("split"))


@pytest.fixture()
def tiny_split() -> SplitData:
    """Return a minimal SplitData with purely numeric features, no CSV needed."""
    rng = np.random.default_rng(7)
    n_train, n_holdout = 40, 20
    X_train = pd.DataFrame(
        {"a": rng.random(n_train).astype(str), "b": rng.random(n_train).astype(str)}
    )
    y_train = pd.Series(rng.integers(0, 2, n_train), name="label")
    X_holdout = pd.DataFrame(
        {"a": rng.random(n_holdout).astype(str), "b": rng.random(n_holdout).astype(str)}
    )
    y_holdout = pd.Series(rng.integers(0, 2, n_holdout), name="label")
    return SplitData(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        orig_index_train=X_train.index,
        orig_index_holdout=X_holdout.index,
        split_fingerprint="test",
    )
