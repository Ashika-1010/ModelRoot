"""M9 tests: hypothesis framework, budget, cache, runner, diagnose."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from modelroot.engine.pipeline import diagnose
from modelroot.experiments.budget import BudgetExceeded, BudgetTracker
from modelroot.experiments.cache import PredictionCache
from modelroot.experiments.interventions import (
    get_intervention,
    intervention,
    list_interventions,
)
from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.hypotheses.registry import clear_registry, get_registry, register_hypothesis
from modelroot.schemas.common import Algorithm, HypothesisType, SplitMode, Status
from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig
from modelroot.schemas.experiment import ExperimentSpec
from modelroot.schemas.hypothesis import Decision, HypothesisResult, TriggerResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: pathlib.Path, n: int = 60) -> AnalysisConfig:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 70, size=n).astype(str),
            "income": rng.integers(30000, 120000, size=n).astype(str),
            "label": ["yes" if rng.random() > 0.5 else "no" for _ in range(n)],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return AnalysisConfig(
        data_path=p,
        target="label",
        positive_class="yes",
        split=SplitConfig(mode=SplitMode.single_file),
        model_spec=ModelSpec(algorithm=Algorithm.logistic_regression, params={}),
        seed=0,
    )


class _FakeHypothesis(Hypothesis):
    """Minimal test hypothesis that always triggers and returns not_triggered."""

    type = HypothesisType.overfitting

    def applicable(self, ctx: HypothesisContext) -> Applicability:
        return Applicability(applicable=True)

    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        return TriggerResult(fired=False, rule_id="FAKE-T1")

    def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list[ExperimentSpec]:
        return []

    def evaluate(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
        results: list,
    ) -> list[HypothesisResult]:
        return [
            HypothesisResult(
                id="fake-001",
                type=HypothesisType.overfitting,
                status=Status.not_triggered,
                evidence_strength=0.0,
                trigger=trigger,
                decision=Decision(
                    rule_id="FAKE-D1",
                    rule_text="Fake rule",
                    rules_profile="full",  # type: ignore[arg-type]
                ),
            )
        ]


# ---------------------------------------------------------------------------
# Budget tracker tests
# ---------------------------------------------------------------------------


def test_budget_tracker_creation() -> None:
    bt = BudgetTracker(budget_s=10.0)
    assert bt.remaining_s > 0
    assert bt.elapsed_s >= 0


def test_budget_tracker_zero_raises() -> None:
    with pytest.raises(ValueError, match="budget_s"):
        BudgetTracker(budget_s=0)


def test_budget_tracker_check_passes() -> None:
    bt = BudgetTracker(budget_s=100.0)
    bt.check()  # Should not raise


def test_budget_tracker_child_capped() -> None:
    bt = BudgetTracker(budget_s=100.0)
    child = bt.child(50.0)
    assert child.remaining_s <= 50.0


def test_budget_exceeded_is_exception() -> None:
    assert issubclass(BudgetExceeded, Exception)


# ---------------------------------------------------------------------------
# Cache tests
# ---------------------------------------------------------------------------


def test_cache_get_missing_returns_none() -> None:
    cache = PredictionCache()
    assert cache.get("nonexistent") is None


def test_cache_put_get() -> None:
    cache = PredictionCache()
    cache.put("key1", "value1")
    assert cache.get("key1") == "value1"
    assert len(cache) == 1


def test_cache_make_key_deterministic() -> None:
    k1 = PredictionCache.make_key("fp1", "fp2", "spec", 0, "macro_f1")
    k2 = PredictionCache.make_key("fp1", "fp2", "spec", 0, "macro_f1")
    assert k1 == k2


def test_cache_make_key_different_inputs() -> None:
    k1 = PredictionCache.make_key("fp1", "fp2", "spec", 0, "macro_f1")
    k2 = PredictionCache.make_key("fp1", "fp2", "spec", 1, "macro_f1")
    assert k1 != k2


def test_cache_clear() -> None:
    cache = PredictionCache()
    cache.put("k", "v")
    cache.clear()
    assert len(cache) == 0


# ---------------------------------------------------------------------------
# Intervention registry tests
# ---------------------------------------------------------------------------


def test_intervention_register_and_get() -> None:
    """Test register and lookup (using a unique name to avoid conflicts)."""

    @intervention("test_m9_dummy")
    def dummy_intervention(ctx: object, params: dict) -> object:
        return ctx

    assert "test_m9_dummy" in list_interventions()
    fn = get_intervention("test_m9_dummy")
    assert callable(fn)


def test_intervention_duplicate_raises() -> None:
    with pytest.raises(ValueError, match="already registered"):

        @intervention("test_m9_dup")
        def f1(ctx: object, params: dict) -> object:
            return ctx

        @intervention("test_m9_dup")
        def f2(ctx: object, params: dict) -> object:
            return ctx


def test_intervention_unknown_raises() -> None:
    with pytest.raises(KeyError):
        get_intervention("nonexistent_intervention")


# ---------------------------------------------------------------------------
# Hypothesis registry tests
# ---------------------------------------------------------------------------


def test_register_and_clear_registry() -> None:
    clear_registry()
    register_hypothesis(_FakeHypothesis)
    assert _FakeHypothesis in get_registry()
    clear_registry()
    assert len(get_registry()) == 0


# ---------------------------------------------------------------------------
# diagnose() tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_diagnose_with_no_hypotheses(tmp_path: pathlib.Path) -> None:
    """diagnose with empty registry returns AssessmentResult + empty hypotheses."""
    config = _make_config(tmp_path)
    clear_registry()
    result = diagnose(config)
    assert result.status == "completed"
    assert len(result.hypotheses) == 0
    assert result.assessment is not None
    assert len(result.warnings) == 0


@pytest.mark.integration
def test_diagnose_with_fake_hypothesis(tmp_path: pathlib.Path) -> None:
    """diagnose with one not_triggered hypothesis returns 1 result."""
    config = _make_config(tmp_path)
    result = diagnose(config, hypotheses=[_FakeHypothesis])
    assert result.status == "completed"
    assert len(result.hypotheses) == 1
    assert result.hypotheses[0].status == Status.not_triggered


@pytest.mark.integration
def test_diagnose_deterministic(tmp_path: pathlib.Path) -> None:
    """Same config + same hypotheses -> same reproducibility_hash."""
    config = _make_config(tmp_path)
    r1 = diagnose(config, hypotheses=[_FakeHypothesis])
    r2 = diagnose(config, hypotheses=[_FakeHypothesis])
    assert r1.reproducibility_hash == r2.reproducibility_hash


@pytest.mark.integration
def test_diagnose_progress_callback(tmp_path: pathlib.Path) -> None:
    """Progress callback receives stage updates."""
    config = _make_config(tmp_path)
    stages: list[tuple[str, float]] = []
    diagnose(config, hypotheses=[], on_progress=lambda s, f: stages.append((s, f)))
    assert len(stages) > 0
    assert stages[-1] == ("done", 1.0)


@pytest.mark.integration
def test_diagnose_hypothesis_exception_contained(tmp_path: pathlib.Path) -> None:
    """Hypothesis that raises is caught as a warning, not a crash."""

    class _CrashingHypothesis(Hypothesis):
        type = HypothesisType.overfitting

        def applicable(self, ctx: HypothesisContext) -> Applicability:
            raise RuntimeError("crash on purpose")

        def trigger(self, ctx: HypothesisContext) -> TriggerResult:
            return TriggerResult(fired=False, rule_id="X")

        def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list:
            return []

        def evaluate(self, ctx: HypothesisContext, trigger: TriggerResult, results: list) -> list:
            return []

    config = _make_config(tmp_path)
    result = diagnose(config, hypotheses=[_CrashingHypothesis])
    assert result.status == "completed"
    assert len(result.warnings) > 0
    assert any("CrashingHypothesis" in w for w in result.warnings)


@pytest.mark.integration
def test_diagnose_not_applicable_skipped(tmp_path: pathlib.Path) -> None:
    """Hypothesis that is not applicable is skipped."""

    class _NotApplicableHypothesis(Hypothesis):
        type = HypothesisType.overfitting

        def applicable(self, ctx: HypothesisContext) -> Applicability:
            return Applicability(applicable=False, reason="test")

        def trigger(self, ctx: HypothesisContext) -> TriggerResult:
            return TriggerResult(fired=False, rule_id="X")

        def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list:
            return []

        def evaluate(self, ctx: HypothesisContext, trigger: TriggerResult, results: list) -> list:
            return []

    config = _make_config(tmp_path)
    result = diagnose(config, hypotheses=[_NotApplicableHypothesis])
    assert len(result.hypotheses) == 0


@pytest.mark.integration
def test_diagnose_run_experiments_false(tmp_path: pathlib.Path) -> None:
    """When run_experiments=False, no experiments are run even if trigger fires."""

    class _TriggeredHypothesis(Hypothesis):
        type = HypothesisType.overfitting

        def applicable(self, ctx: HypothesisContext) -> Applicability:
            return Applicability(applicable=True)

        def trigger(self, ctx: HypothesisContext) -> TriggerResult:
            return TriggerResult(fired=True, rule_id="T")

        def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list:
            return [
                ExperimentSpec(
                    id="exp-test-001",
                    hypothesis_id="hyp-test",
                    name="test",
                    intervention="test_m9_dummy",
                    metric="macro_f1",
                    seed=0,
                    budget_s=10.0,
                )
            ]

        def evaluate(self, ctx: HypothesisContext, trigger: TriggerResult, results: list) -> list:
            return [
                HypothesisResult(
                    id="hyp-test",
                    type=HypothesisType.overfitting,
                    status=Status.not_triggered,
                    evidence_strength=0.0,
                    trigger=trigger,
                    decision=Decision(
                        rule_id="D",
                        rule_text="rule",
                        rules_profile="full",  # type: ignore[arg-type]
                    ),
                )
            ]

    config = _make_config(tmp_path)
    config_no_exp = AnalysisConfig(
        data_path=config.data_path,
        target=config.target,
        positive_class=config.positive_class,
        split=config.split,
        model_spec=config.model_spec,
        seed=config.seed,
        run_experiments=False,
    )
    result = diagnose(config_no_exp, hypotheses=[_TriggeredHypothesis])
    assert len(result.experiments) == 0


# ---------------------------------------------------------------------------
# AnalysisResult structure tests
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_diagnose_result_has_fingerprints(tmp_path: pathlib.Path) -> None:
    config = _make_config(tmp_path)
    result = diagnose(config)
    assert len(result.dataset_fingerprint) == 64
    assert len(result.split_fingerprint) == 64
    assert len(result.reproducibility_hash) == 64


@pytest.mark.integration
def test_diagnose_result_has_timings(tmp_path: pathlib.Path) -> None:
    config = _make_config(tmp_path)
    result = diagnose(config)
    assert "prepare" in result.timings
    assert "assess" in result.timings
    assert "total" in result.timings
