"""Tests for M5 baseline and dummy baseline (T5-08 to T5-18)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from modelroot.assessment.baseline import (
    BaselineRun,
    PredictionSet,
    dummy_baseline,
    fit_baseline,
)
from modelroot.data.loader import load_csv
from modelroot.data.split import SplitData, make_split
from modelroot.data.validate import validate_for_analysis
from modelroot.models.registry import get_model
from modelroot.schemas.common import Algorithm, SplitMode
from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig
from modelroot.utils.seeds import SeedManager


def _make_split(
    tmp_path: pathlib.Path, n: int = 60, algo: Algorithm = Algorithm.logistic_regression
) -> tuple[SplitData, AnalysisConfig]:
    rng = np.random.default_rng(42)
    ages = rng.integers(20, 70, size=n).astype(str)
    incomes = rng.integers(30000, 120000, size=n).astype(str)
    labels = ["yes" if rng.random() > 0.5 else "no" for _ in range(n)]
    df = pd.DataFrame({"age": ages, "income": incomes, "label": labels})
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    df_loaded = load_csv(p)
    config = AnalysisConfig(
        data_path=p,
        target="label",
        positive_class="yes",
        split=SplitConfig(mode=SplitMode.single_file),
        model_spec=ModelSpec(algorithm=algo, params={}),
        seed=0,
    )
    validated = validate_for_analysis(df_loaded, config)
    sm = SeedManager(0)
    split = make_split(validated, config, seed=sm.spawn("split"))
    return split, config


# T5-08: PredictionSet is frozen
def test_prediction_set_frozen() -> None:
    ps = PredictionSet(
        index=pd.Index([0, 1]),
        y_true=np.array([0, 1]),
        y_pred=np.array([0, 1]),
        y_proba=np.array([0.1, 0.9]),
    )
    with pytest.raises((AttributeError, TypeError)):
        ps.y_true = np.array([1, 0])  # type: ignore[misc]


# T5-09: BaselineRun is frozen
def test_baseline_run_frozen(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    run = fit_baseline(adapter, split)
    assert isinstance(run, BaselineRun)
    with pytest.raises((AttributeError, TypeError)):
        run.notes = []  # type: ignore[misc]


# T5-10: fit_baseline does not mutate original adapter
def test_fit_baseline_does_not_mutate_adapter(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    original_pipeline = adapter._pipeline
    _ = fit_baseline(adapter, split)
    # Original adapter's pipeline should still be None (unfitted)
    assert adapter._pipeline is original_pipeline


# T5-11: fit_baseline produces train and holdout metrics
def test_fit_baseline_produces_metrics(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    run = fit_baseline(adapter, split)
    splits_in_metrics = {m.split for m in run.metrics}
    assert "train" in splits_in_metrics
    assert "holdout" in splits_in_metrics
    # All metric values in [0, 1]
    for m in run.metrics:
        assert 0.0 <= m.value <= 1.0, f"{m.name} on {m.split} = {m.value}"


# T5-12: Determinism — same seed produces identical results
def test_fit_baseline_deterministic(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter1 = get_model(config.model_spec, col_types, seed=42)
    adapter2 = get_model(config.model_spec, col_types, seed=42)
    run1 = fit_baseline(adapter1, split)
    run2 = fit_baseline(adapter2, split)
    np.testing.assert_array_equal(run1.holdout_pred.y_pred, run2.holdout_pred.y_pred)
    np.testing.assert_allclose(run1.holdout_pred.y_proba, run2.holdout_pred.y_proba)


# T5-13: All four algorithms fit and predict
@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.logistic_regression,
        Algorithm.decision_tree,
        Algorithm.random_forest,
        Algorithm.hist_gradient_boosting,
    ],
)
def test_all_algorithms_fit_baseline(tmp_path: pathlib.Path, algo: Algorithm) -> None:
    split, config = _make_split(tmp_path, algo=algo)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    run = fit_baseline(adapter, split)
    assert len(run.holdout_pred.y_pred) == len(split.y_holdout)
    proba = run.holdout_pred.y_proba
    assert proba.shape == (len(split.y_holdout),)
    assert np.all(proba >= 0) and np.all(proba <= 1)


# T5-14: Probabilities are 1-D class-1 scores
def test_probabilities_are_1d(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    run = fit_baseline(adapter, split)
    assert run.holdout_pred.y_proba.ndim == 1


# T5-15: Indices match orig_index
def test_indices_preserved(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    run = fit_baseline(adapter, split)
    np.testing.assert_array_equal(run.train_pred.index, split.orig_index_train)
    np.testing.assert_array_equal(run.holdout_pred.index, split.orig_index_holdout)


# T5-16: dummy_baseline — majority class prediction
def test_dummy_baseline_majority_class(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    metrics = dummy_baseline(split)
    acc = next((m.value for m in metrics if m.name == "accuracy"), None)
    y_holdout = np.asarray(split.y_holdout.values)
    majority = 1 if np.sum(split.y_train.values == 1) > len(split.y_train) / 2 else 0
    expected_acc = float(np.mean(y_holdout == majority))
    assert acc == pytest.approx(expected_acc, abs=1e-9)


# T5-17: dummy_baseline tie-breaks toward class 0
def test_dummy_baseline_tie_toward_zero() -> None:
    # Perfectly balanced training — tie → majority class = 0
    y_train = pd.Series([0, 1, 0, 1, 0, 1, 0, 1, 0, 1])
    y_holdout = pd.Series([0, 1, 0, 1, 0])
    X_train = pd.DataFrame({"a": ["1"] * 10})
    X_holdout = pd.DataFrame({"a": ["1"] * 5})
    split = SplitData(
        X_train=X_train,
        y_train=y_train,
        X_holdout=X_holdout,
        y_holdout=y_holdout,
        orig_index_train=X_train.index,
        orig_index_holdout=X_holdout.index,
        split_fingerprint="tie-test",
    )
    metrics = dummy_baseline(split)
    acc = next((m.value for m in metrics if m.name == "accuracy"), None)
    # majority = 0, so accuracy = fraction of holdout that is 0 = 3/5
    assert acc == pytest.approx(3 / 5)


# T5-18: end-to-end M2->M5 integration
def test_end_to_end_m2_m5(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path, n=80)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    run = fit_baseline(adapter, split)
    dummy = dummy_baseline(split)

    # Check structure
    assert isinstance(run, BaselineRun)
    assert len(dummy) > 0
    # At least accuracy should always be defined
    assert any(m.name == "accuracy" for m in run.metrics)
    assert any(m.name == "accuracy" for m in dummy)
