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
