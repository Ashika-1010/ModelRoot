"""Tests for benchmark data contracts (M12)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelroot_bench.scenario import (
    BenchmarkRecord,
    GroundTruth,
    InjectedFailure,
    InjectionSpec,
    ScenarioResult,
    ScenarioSpec,
)


def test_scenario_spec_valid() -> None:
    spec = ScenarioSpec(
        id="syn_cc_rf_0",
        dataset="synthetic_easy",
        model_spec={"algorithm": "random_forest", "params": {}},
        scenario_class="clean_control",
        injections=[],
        seed=42,
        role="dev",
    )
    assert spec.id == "syn_cc_rf_0"
    assert spec.scenario_class == "clean_control"
    assert spec.seed == 42


def test_scenario_spec_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        ScenarioSpec(
            id="bad_spec",
            dataset="synthetic_easy",
            model_spec={"algorithm": "random_forest", "params": {}},
            scenario_class="clean_control",
            seed=0,
            unknown_field=123,  # type: ignore[call-arg]
        )


def test_ground_truth_contract() -> None:
    gt = GroundTruth(
        injected=[InjectedFailure(label="overfitting", severity="S2")],
        distractors=["label_noise"],
        impact=0.08,
        no_effect=False,
    )
    assert len(gt.injected) == 1
    assert gt.injected[0].label == "overfitting"
    assert gt.impact == 0.08
    assert not gt.no_effect


def test_scenario_result_contract() -> None:
    res = ScenarioResult(
        scenario_id="syn_of_rf_0",
        statuses={"overfitting": "supported"},
        top1="overfitting",
        runtime_s=1.23,
        analysis_hash="deadbeef" * 8,
        control_statuses={"overfitting": "not_triggered"},
        effects={"overfitting": 0.05},
        ranking=["overfitting"],
    )
    assert res.top1 == "overfitting"
    assert res.statuses["overfitting"] == "supported"
    assert res.effects["overfitting"] == 0.05


def test_benchmark_record_contract() -> None:
    spec = ScenarioSpec(
        id="s1",
        dataset="d1",
        model_spec={"algorithm": "decision_tree", "params": {}},
        scenario_class="single_failure",
        injections=[InjectionSpec(label="overfitting", severity="S1")],
        seed=1,
        role="dev",
    )
    gt = GroundTruth(
        injected=[InjectedFailure(label="overfitting", severity="S1")],
        impact=0.05,
    )
    rec = BenchmarkRecord(spec=spec, ground_truth=gt)
    assert rec.result is None
    dumped = rec.model_dump()
    assert dumped["spec"]["id"] == "s1"
    assert dumped["ground_truth"]["injected"][0]["label"] == "overfitting"
