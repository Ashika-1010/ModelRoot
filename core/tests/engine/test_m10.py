"""M10 tests: rule engine, decision protocol, evidence, ranking, wording."""

from __future__ import annotations

import pytest

from modelroot.config.thresholds import Thresholds
from modelroot.evidence.rules import (
    RULE_SET_VERSION,
    Criterion,
    Rule,
    RuleSet,
    check_wording,
    clear_rule_registry,
    decide_intervention_effect,
    evidence_strength,
    get_rule_registry,
    make_evidence_id,
    rank,
    register_rule,
    severity,
)
from modelroot.schemas.common import RulesProfile, Severity, Status
from modelroot.schemas.hypothesis import Decision, HypothesisResult, TriggerResult

# ---------------------------------------------------------------------------
# Rule / Criterion / RuleSet
# ---------------------------------------------------------------------------


def test_rule_set_version() -> None:
    assert RULE_SET_VERSION == "1.0"


def test_criterion_frozen() -> None:
    c = Criterion(id="c1", weight=2.0, description="test")
    assert c.id == "c1"
    assert c.weight == 2.0


def test_rule_set_add_get() -> None:
    rs = RuleSet()
    r = Rule(id="R1", text="test rule", criteria=[Criterion(id="c1")])
    rs.add(r)
    assert rs.get("R1") is r
    assert rs.get("nonexistent") is None


def test_rule_set_to_json() -> None:
    rs = RuleSet()
    rs.add(Rule(id="R1", text="rule 1", criteria=[Criterion(id="c1", weight=1.0)]))
    j = rs.to_json()
    assert "R1" in j
    assert "c1" in j


def test_rule_set_to_markdown() -> None:
    rs = RuleSet()
    rs.add(Rule(id="R1", text="rule 1", criteria=[Criterion(id="c1")]))
    md = rs.to_markdown()
    assert "R1" in md
    assert "rule 1" in md


def test_register_and_clear_rule_registry() -> None:
    clear_rule_registry()
    r = Rule(id="R1", text="test")
    register_rule(r)
    assert r in get_rule_registry()
    clear_rule_registry()
    assert len(get_rule_registry()) == 0


# ---------------------------------------------------------------------------
# Decision protocol — full profile
# ---------------------------------------------------------------------------


def _thresholds() -> Thresholds:
    return Thresholds()


def test_decide_supported() -> None:
    t = _thresholds()
    status, criteria = decide_intervention_effect(
        delta=0.05,
        ci_low=0.01,
        ci_high=0.09,
        gate_ok=True,
        extra_criteria={"x": True},
        thresholds=t,
        profile=RulesProfile.full,
    )
    assert status == Status.supported


def test_decide_weakly_supported() -> None:
    t = _thresholds()
    status, _ = decide_intervention_effect(
        delta=0.015,
        ci_low=-0.01,
        ci_high=0.04,
        gate_ok=True,
        extra_criteria={},
        thresholds=t,
        profile=RulesProfile.full,
    )
    # delta >= delta_min/2 (0.01) and delta > 0 -> weakly_supported
    assert status == Status.weakly_supported


def test_decide_rejected_equivalence() -> None:
    t = _thresholds()
    status, _ = decide_intervention_effect(
        delta=0.001,
        ci_low=-0.01,
        ci_high=0.015,
        gate_ok=True,
        extra_criteria={},
        thresholds=t,
        profile=RulesProfile.full,
    )
    # ci_high < delta_min (0.02) -> rejected
    assert status == Status.rejected


def test_decide_inconclusive_gate_fail() -> None:
    t = _thresholds()
    status, _ = decide_intervention_effect(
        delta=0.05,
        ci_low=0.01,
        ci_high=0.09,
        gate_ok=False,
        extra_criteria={},
        thresholds=t,
        profile=RulesProfile.full,
    )
    assert status == Status.inconclusive


def test_decide_extra_criteria_not_met() -> None:
    t = _thresholds()
    status, _ = decide_intervention_effect(
        delta=0.05,
        ci_low=0.01,
        ci_high=0.09,
        gate_ok=True,
        extra_criteria={"x": False},
        thresholds=t,
        profile=RulesProfile.full,
    )
    assert status != Status.supported


# ---------------------------------------------------------------------------
# Decision protocol — naive profile
# ---------------------------------------------------------------------------


def test_decide_naive_supported() -> None:
    t = _thresholds()
    status, _ = decide_intervention_effect(
        delta=0.05,
        ci_low=0.0,
        ci_high=0.1,
        gate_ok=True,
        extra_criteria={},
        thresholds=t,
        profile=RulesProfile.naive,
    )
    assert status == Status.supported


def test_decide_naive_rejected() -> None:
    t = _thresholds()
    status, _ = decide_intervention_effect(
        delta=0.001,
        ci_low=0.0,
        ci_high=0.1,
        gate_ok=True,
        extra_criteria={},
        thresholds=t,
        profile=RulesProfile.naive,
    )
    assert status == Status.rejected


# ---------------------------------------------------------------------------
# Evidence strength
# ---------------------------------------------------------------------------


def test_evidence_strength_all_met() -> None:
    r = Rule(
        id="R", text="t", criteria=[Criterion(id="c1", weight=1.0), Criterion(id="c2", weight=2.0)]
    )
    s = evidence_strength(r, {"c1": True, "c2": True})
    assert s == pytest.approx(1.0)


def test_evidence_strength_none_met() -> None:
    r = Rule(
        id="R", text="t", criteria=[Criterion(id="c1", weight=1.0), Criterion(id="c2", weight=2.0)]
    )
    s = evidence_strength(r, {"c1": False, "c2": False})
    assert s == pytest.approx(0.0)


def test_evidence_strength_partial() -> None:
    r = Rule(
        id="R", text="t", criteria=[Criterion(id="c1", weight=1.0), Criterion(id="c2", weight=3.0)]
    )
    s = evidence_strength(r, {"c1": True, "c2": False})
    assert s == pytest.approx(0.25)


def test_evidence_strength_in_01() -> None:
    r = Rule(id="R", text="t", criteria=[Criterion(id="c1", weight=1.0)])
    s = evidence_strength(r, {"c1": True})
    assert 0.0 <= s <= 1.0


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


def test_severity_high() -> None:
    assert severity(0.06, _thresholds()) == Severity.high


def test_severity_medium() -> None:
    assert severity(0.03, _thresholds()) == Severity.medium


def test_severity_low() -> None:
    assert severity(0.01, _thresholds()) == Severity.low


def test_severity_negative_delta() -> None:
    assert severity(-0.06, _thresholds()) == Severity.high


def test_severity_non_finite() -> None:
    assert severity(float("inf"), _thresholds()) is None


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


def _make_hyp_result(
    hyp_id: str, status: Status, delta: float = 0.0, strength: float = 0.0
) -> HypothesisResult:
    return HypothesisResult(
        id=hyp_id,
        type="overfitting",  # type: ignore[arg-type]
        status=status,
        evidence_strength=strength,
        trigger=TriggerResult(fired=False, rule_id="X"),
        decision=Decision(rule_id="D", rule_text="t", rules_profile="full"),  # type: ignore[arg-type]
    )


def test_rank_supported_before_inconclusive() -> None:
    results = [
        _make_hyp_result("h1", Status.inconclusive),
        _make_hyp_result("h2", Status.supported),
    ]
    ranked = rank(results)
    assert ranked[0].id == "h2"


def test_rank_by_delta_descending() -> None:
    from modelroot.schemas.hypothesis import EffectResult

    r1 = _make_hyp_result("h1", Status.supported, strength=0.5)
    r1 = HypothesisResult(
        id="h1",
        type="overfitting",
        status=Status.supported,  # type: ignore[arg-type]
        evidence_strength=0.5,
        trigger=TriggerResult(fired=False, rule_id="X"),
        decision=Decision(rule_id="D", rule_text="t", rules_profile="full"),  # type: ignore[arg-type]
        effect=EffectResult(
            metric="macro_f1",
            baseline=0.5,
            intervention=0.6,
            delta=0.1,
            ci_low=0.0,
            ci_high=0.2,
            n=100,
        ),
    )
    r2 = HypothesisResult(
        id="h2",
        type="overfitting",
        status=Status.supported,  # type: ignore[arg-type]
        evidence_strength=0.3,
        trigger=TriggerResult(fired=False, rule_id="X"),
        decision=Decision(rule_id="D", rule_text="t", rules_profile="full"),  # type: ignore[arg-type]
        effect=EffectResult(
            metric="macro_f1",
            baseline=0.5,
            intervention=0.55,
            delta=0.05,
            ci_low=0.0,
            ci_high=0.1,
            n=100,
        ),
    )
    ranked = rank([r2, r1])
    assert ranked[0].id == "h1"  # higher delta first


def test_rank_stable_under_permutation() -> None:
    results = [
        _make_hyp_result("a", Status.supported, strength=0.5),
        _make_hyp_result("b", Status.supported, strength=0.5),
        _make_hyp_result("c", Status.inconclusive, strength=0.5),
    ]
    r1 = rank(results)
    r2 = rank(list(reversed(results)))
    assert [r.id for r in r1] == [r.id for r in r2]


# ---------------------------------------------------------------------------
# Wording validation
# ---------------------------------------------------------------------------


def test_wording_banned_phrase_detected() -> None:
    violations = check_wording("This proves the issue", Status.supported)
    assert len(violations) > 0


def test_wording_clean_text() -> None:
    violations = check_wording("The evidence supports this hypothesis.", Status.supported)
    assert len(violations) == 0


def test_wording_all_banned() -> None:
    for phrase in ["proves", "proven", "confirmed", "definitely", "root cause is", "caused by"]:
        violations = check_wording(f"This {phrase} the problem.", Status.supported)
        assert len(violations) > 0


# ---------------------------------------------------------------------------
# Evidence ID builder
# ---------------------------------------------------------------------------


def test_make_evidence_id_stable() -> None:
    id1 = make_evidence_id("gap", 0.19, "metric_difference", None)
    id2 = make_evidence_id("gap", 0.19, "metric_difference", None)
    assert id1 == id2
    assert id1.startswith("ev_")
    assert len(id1) == 11  # "ev_" + 8 hex chars


def test_make_evidence_id_different_inputs() -> None:
    id1 = make_evidence_id("gap", 0.19, "metric_difference", None)
    id2 = make_evidence_id("gap", 0.20, "metric_difference", None)
    assert id1 != id2


# ---------------------------------------------------------------------------
# Integration: diagnose() applies M10 finalization
# ---------------------------------------------------------------------------


def test_diagnose_applies_finalization(tmp_path: object) -> None:
    """Regression: diagnose() must apply M10 finalize (ranking + severity)."""
    import pathlib

    import numpy as np
    import pandas as pd

    from modelroot.engine.pipeline import diagnose
    from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
    from modelroot.schemas.common import Algorithm, HypothesisType, SplitMode, Status
    from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig
    from modelroot.schemas.experiment import ExperimentSpec
    from modelroot.schemas.hypothesis import (
        Decision,
        EffectResult,
        HypothesisResult,
        TriggerResult,
    )

    # Create a small CSV
    rng = np.random.default_rng(42)
    n = 60
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 70, size=n).astype(str),
            "income": rng.integers(30000, 120000, size=n).astype(str),
            "label": ["yes" if rng.random() > 0.5 else "no" for _ in range(n)],
        }
    )
    p = pathlib.Path(str(tmp_path)) / "data.csv"  # type: ignore[arg-type]
    df.to_csv(p, index=False)

    config = AnalysisConfig(
        data_path=p,
        target="label",
        positive_class="yes",
        split=SplitConfig(mode=SplitMode.single_file),
        model_spec=ModelSpec(algorithm=Algorithm.logistic_regression, params={}),
        seed=0,
    )

    class _SupportedHyp(Hypothesis):
        type = HypothesisType.overfitting

        def applicable(self, ctx: HypothesisContext) -> Applicability:
            return Applicability(applicable=True)

        def trigger(self, ctx: HypothesisContext) -> TriggerResult:
            return TriggerResult(fired=True, rule_id="T1")

        def plan(self, ctx: HypothesisContext, trigger: TriggerResult) -> list[ExperimentSpec]:
            return []

        def evaluate(
            self, ctx: HypothesisContext, trigger: TriggerResult, results: list
        ) -> list[HypothesisResult]:
            return [
                HypothesisResult(
                    id="h-supported",
                    type=HypothesisType.overfitting,
                    status=Status.supported,
                    evidence_strength=0.8,
                    trigger=trigger,
                    decision=Decision(
                        rule_id="D1",
                        rule_text="test",
                        rules_profile="full",  # type: ignore[arg-type]
                    ),
                    effect=EffectResult(
                        metric="macro_f1",
                        baseline=0.5,
                        intervention=0.6,
                        delta=0.1,
                        ci_low=0.01,
                        ci_high=0.19,
                        n=100,
                    ),
                )
            ]

    result = diagnose(config, hypotheses=[_SupportedHyp])

    # Severity should have been assigned by finalize
    assert len(result.hypotheses) == 1
    h = result.hypotheses[0]
    assert h.id == "h-supported"
    assert h.severity is not None  # finalize assigned it
    assert h.severity.value == "high"  # delta=0.1 >= 0.05

    # rule_set_version should use the M10 constant
    assert result.rule_set_version == "1.0"
