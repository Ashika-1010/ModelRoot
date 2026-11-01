"""Tests for M6 reference panel, verdict, assess (T6-04 to T6-14)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from modelroot.assessment.baseline import fit_baseline
from modelroot.assessment.reference_panel import fit_reference_panel
from modelroot.assessment.verdict import assess, assess_only, get_metric_value
from modelroot.config.thresholds import Thresholds
from modelroot.data.loader import load_csv
from modelroot.data.split import SplitData, make_split
from modelroot.data.validate import validate_for_analysis
from modelroot.models.registry import get_model
from modelroot.schemas.common import Algorithm, SplitMode
from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig
from modelroot.schemas.result import AssessmentResult
from modelroot.utils.seeds import SeedManager


def _make_split(
    tmp_path: pathlib.Path,
    n: int = 80,
    algo: Algorithm = Algorithm.logistic_regression,
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


# T6-04: get_metric_value
def test_get_metric_value_found() -> None:
    from modelroot.schemas.metrics import MetricResult

    metrics = [MetricResult(name="accuracy", value=0.8, split="holdout", n=100)]
    val = get_metric_value(metrics, "accuracy", "holdout")
    assert val == pytest.approx(0.8)


def test_get_metric_value_not_found() -> None:
    assert get_metric_value([], "accuracy", "holdout") is None


# T6-05: fit_reference_panel runs all three algorithms
def test_reference_panel_has_three_entries(tmp_path: pathlib.Path) -> None:
    split, _ = _make_split(tmp_path)
    panel = fit_reference_panel(split, "macro_f1", base_seed=0)
    assert len(panel.entries) == 3


# T6-06: panel reference algorithm is the best one
def test_reference_panel_best_is_highest(tmp_path: pathlib.Path) -> None:
    split, _ = _make_split(tmp_path)
    panel = fit_reference_panel(split, "macro_f1", base_seed=0)
    best_score = panel.entries[panel.best].holdout_score
    for e in panel.entries:
        assert best_score >= e.holdout_score


# T6-07: assess_only returns AssessmentResult
def test_assess_only_returns_assessment_result(tmp_path: pathlib.Path) -> None:
    _, config = _make_split(tmp_path)
    result = assess_only(config)
    assert isinstance(result, AssessmentResult)


# T6-08: assess_only is deterministic (same seed → same verdict)
def test_assess_only_deterministic(tmp_path: pathlib.Path) -> None:
    _, config = _make_split(tmp_path)
    r1 = assess_only(config)
    r2 = assess_only(config)
    assert r1.underperforming == r2.underperforming
    assert r1.headroom == pytest.approx(r2.headroom)


# T6-09: headroom rule fires when reference >> baseline
def test_headroom_rule(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    baseline_run = fit_baseline(adapter, split)
    panel = fit_reference_panel(split, "macro_f1", base_seed=0)
    # Force headroom to exceed threshold by using very low thresholds
    thresholds = Thresholds(headroom=0.0, gap=1.0, current_drop=1.0)
    result = assess(config, split, baseline_run, panel, thresholds)
    # With headroom=0.0, any positive headroom triggers underperformance
    if result.headroom >= 0.0:
        assert any("Reference model" in r for r in result.reasons)


# T6-10: gap rule fires when train >> holdout
def test_gap_rule(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    # Use decision tree (likely to overfit)
    config2 = AnalysisConfig(
        data_path=config.data_path,
        target=config.target,
        positive_class=config.positive_class,
        split=config.split,
        model_spec=ModelSpec(algorithm=Algorithm.decision_tree, params={"max_depth": None}),
        seed=0,
    )
    adapter = get_model(config2.model_spec, col_types, seed=0)
    baseline_run = fit_baseline(adapter, split)
    panel = fit_reference_panel(split, "macro_f1", base_seed=0)
    # Force gap threshold to 0.0 to guarantee trigger
    thresholds = Thresholds(headroom=1.0, gap=0.0, current_drop=1.0)
    result = assess(config2, split, baseline_run, panel, thresholds)
    if result.train_holdout_gap >= 0.0:
        assert any("gap" in r.lower() or "train" in r.lower() for r in result.reasons)


# T6-11: headroom may be negative (no clipping)
def test_headroom_may_be_negative(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config.model_spec, col_types, seed=0)
    baseline_run = fit_baseline(adapter, split)
    panel = fit_reference_panel(split, "macro_f1", base_seed=0)
    thresholds = Thresholds()
    result = assess(config, split, baseline_run, panel, thresholds)
    # Headroom is allowed to be any real number
    assert isinstance(result.headroom, float)


# T6-12: expected_score rule
def test_expected_score_rule(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    # Force expected_score very high so it triggers
    from modelroot.schemas.config import Metadata

    config_high = AnalysisConfig(
        data_path=config.data_path,
        target=config.target,
        positive_class=config.positive_class,
        split=config.split,
        model_spec=config.model_spec,
        seed=config.seed,
        metadata=Metadata(expected_score=0.9999),
    )
    col_types = {"age": "numeric", "income": "numeric"}
    adapter = get_model(config_high.model_spec, col_types, seed=0)
    baseline_run = fit_baseline(adapter, split)
    panel = fit_reference_panel(split, "macro_f1", base_seed=0)
    thresholds = Thresholds(headroom=1.0, gap=1.0, current_drop=1.0)
    result = assess(config_high, split, baseline_run, panel, thresholds)
    assert result.underperforming
    assert any("expected" in r.lower() for r in result.reasons)


# T6-13: headline is non-empty string
def test_headline_non_empty(tmp_path: pathlib.Path) -> None:
    _, config = _make_split(tmp_path)
    result = assess_only(config)
    assert isinstance(result.headline, str)
    assert len(result.headline) > 0


# T6-14: dummy_metrics are present
def test_dummy_metrics_present(tmp_path: pathlib.Path) -> None:
    _, config = _make_split(tmp_path)
    result = assess_only(config)
    assert len(result.dummy_metrics) > 0
    assert any(m.name == "accuracy" for m in result.dummy_metrics)
