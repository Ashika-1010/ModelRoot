"""Tests for benchmark runner skeleton (M12)."""

from __future__ import annotations

from modelroot_bench.runner import run_scenario
from modelroot_bench.scenario import ScenarioResult, ScenarioSpec


def test_run_scenario_with_mock_builder() -> None:
    spec = ScenarioSpec(
        id="test_scen",
        dataset="synth",
        model_spec={"algorithm": "random_forest", "params": {}},
        scenario_class="clean_control",
        seed=1,
    )

    def mock_build(s: ScenarioSpec) -> ScenarioResult:
        return ScenarioResult(
            scenario_id=s.id,
            statuses={"overfitting": "not_triggered"},
            runtime_s=0.12,
        )

    res = run_scenario(spec, mock_build)
    assert res.scenario_id == "test_scen"
    assert res.statuses["overfitting"] == "not_triggered"
    assert res.runtime_s == 0.12


def test_run_scenario_with_dict_builder() -> None:
    spec = ScenarioSpec(
        id="test_scen_dict",
        dataset="synth",
        model_spec={"algorithm": "decision_tree", "params": {}},
        scenario_class="single_failure",
        seed=2,
    )

    def mock_dict_build(s: ScenarioSpec) -> dict[str, object]:
        return {
            "statuses": {"overfitting": "supported"},
            "top1": "overfitting",
            "effects": {"overfitting": 0.04},
        }

    res = run_scenario(spec, mock_dict_build)
    assert res.scenario_id == "test_scen_dict"
    assert res.statuses["overfitting"] == "supported"
    assert res.top1 == "overfitting"
    assert res.effects["overfitting"] == 0.04
    assert res.runtime_s > 0.0


def test_run_scenario_with_analysis_result_subtypes() -> None:
    """Verify that AnalysisResult with sub-claims gets namespaced as type:subtype."""
    from unittest.mock import MagicMock

    from modelroot_bench.scoring import _aggregate_statuses

    spec = ScenarioSpec(
        id="test_scen_subtypes",
        dataset="synth",
        model_spec={"algorithm": "random_forest", "params": {}},
        scenario_class="single_failure",
        seed=3,
    )

    h_distortion = MagicMock()
    h_distortion.type.value = "imbalance"
    h_distortion.subtype = "distortion"
    h_distortion.status.value = "supported"
    h_distortion.effect.delta = 0.05

    h_harm = MagicMock()
    h_harm.type.value = "imbalance"
    h_harm.subtype = "harm"
    h_harm.status.value = "weakly_supported"
    h_harm.effect.delta = 0.03

    h_leak = MagicMock()
    h_leak.type.value = "leakage"
    h_leak.subtype = "duplicate"
    h_leak.status.value = "supported"
    h_leak.effect.delta = 0.08

    mock_analysis = MagicMock()
    mock_analysis.hypotheses = [h_distortion, h_harm, h_leak]
    mock_analysis.reproducibility_hash = "mock_hash_123"

    res = run_scenario(spec, lambda _: mock_analysis)
    assert res.scenario_id == "test_scen_subtypes"
    assert res.statuses["imbalance:distortion"] == "supported"
    assert res.statuses["imbalance:harm"] == "weakly_supported"
    assert res.statuses["leakage:duplicate"] == "supported"

    # Aggregated statuses should contain type-level entries
    agg = _aggregate_statuses(res.statuses)
    assert agg["imbalance"] == "supported"  # best status between supported and weakly_supported
    assert agg["leakage"] == "supported"
    assert agg["imbalance:distortion"] == "supported"
