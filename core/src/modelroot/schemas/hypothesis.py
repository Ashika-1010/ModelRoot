"""Hypothesis contracts: TriggerResult, EffectResult, ExplanationOutput,
Decision, and HypothesisResult.

A HypothesisResult is the complete verdict for one hypothesis (or
sub-hypothesis) after triggers, experiments, and the rule engine have run.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelroot.schemas.common import (
    HypothesisType,
    RulesProfile,
    Severity,
    Status,
)

# ---------------------------------------------------------------------------
# Valid subtype values per hypothesis type
# ---------------------------------------------------------------------------

#: Mapping from HypothesisType to the set of allowed subtype strings.
#: ``None`` in the set means the subtype field may be omitted / null.
_VALID_SUBTYPES: dict[HypothesisType, set[str | None]] = {
    HypothesisType.overfitting: {None},
    HypothesisType.imbalance: {"distortion", "harm"},
    HypothesisType.leakage: {"duplicate", "identifier", "target_proxy"},
    HypothesisType.subgroup: {"representation_related", "intrinsic_or_unresolved", None},
    HypothesisType.shift: {None},
}


# ---------------------------------------------------------------------------
# TriggerResult
# ---------------------------------------------------------------------------


class TriggerResult(BaseModel):
    """Outcome of the cheap diagnostic trigger check for a hypothesis.

    Attributes:
        fired: Whether the trigger fired (i.e. experiments should be run).
        rule_id: Identifier of the trigger rule that was evaluated.
        values: Measured diagnostic quantities used by the trigger decision.
    """

    model_config = ConfigDict(extra="forbid")

    fired: bool
    rule_id: str
    values: dict[str, float | str | bool | int | None] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# EffectResult
# ---------------------------------------------------------------------------


class EffectResult(BaseModel):
    """Summary of the primary measured effect for a hypothesis.

    Attributes:
        metric: Name of the metric.
        baseline: Baseline model metric value.
        intervention: Intervention model metric value.
        delta: ``intervention - baseline``.
        ci_low: Lower bound of the paired bootstrap CI on delta.
        ci_high: Upper bound of the paired bootstrap CI on delta.
        n: Evaluation-set size.

    Raises:
        ValueError: If ``ci_low > ci_high``.
    """

    model_config = ConfigDict(extra="forbid")

    metric: str
    baseline: float
    intervention: float
    delta: float
    ci_low: float
    ci_high: float
    n: int

    @model_validator(mode="after")
    def _ci_ordering(self) -> EffectResult:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")
        return self


# ---------------------------------------------------------------------------
# ExplanationOutput
# ---------------------------------------------------------------------------


class ExplanationOutput(BaseModel):
    """Natural-language explanation generated for a hypothesis result.

    Attributes:
        source: How the explanation was produced — ``"template"`` for the
            deterministic template path, ``"llm"`` for the validated LLM path.
        text: The explanation text.
        provenance: Optional provenance string (e.g. model/prompt version)
            when ``source`` is ``"llm"``.
    """

    model_config = ConfigDict(extra="forbid")

    source: str
    text: str
    provenance: str | None = None


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


class Decision(BaseModel):
    """Record of which decision rule was applied and which criteria were met.

    Attributes:
        rule_id: Identifier of the versioned decision rule.
        rule_text: Human-readable description of the rule.
        rules_profile: Which rules profile was active when the decision was made.
        criteria_met: Map from criterion identifier to whether it was satisfied.
    """

    model_config = ConfigDict(extra="forbid")

    rule_id: str
    rule_text: str
    rules_profile: RulesProfile
    criteria_met: dict[str, bool] = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# HypothesisResult
# ---------------------------------------------------------------------------


class HypothesisResult(BaseModel):
    """Complete verdict for one hypothesis (or sub-hypothesis).

    Attributes:
        id: Non-empty unique identifier for this result.
        type: Which of the five hypothesis types this result belongs to.
        subtype: Sub-claim within the hypothesis type, or ``None``.
            Allowed values depend on ``type`` — see the master plan §3.2.
        status: Verdict assigned by the rule engine.
        evidence_strength: Weighted fraction of satisfied decision criteria
            in [0, 1].  For ranking only; not a calibrated probability.
        severity: Estimated impact magnitude (only set when ``status`` is
            ``supported`` or ``weakly_supported``).
        priority_rank: Position in the overall ranking of hypotheses.
        trigger: Outcome of the trigger check.
        diagnostic_evidence_ids: IDs of ``Evidence`` items from diagnostics.
        experiment_ids: IDs of ``ExperimentSpec`` items that were run.
        effect: Summary of the primary measured effect, if available.
        decision: The rule and criteria that produced ``status``.
        confounders_checked: List of confounder names that were assessed.
        caveats: Free-text caveats about this verdict.
        recommended_actions: Suggested next steps for the user.
        explanation: Natural-language explanation, if ``explain=True``.

    Raises:
        ValueError: If ``id`` is empty, ``evidence_strength`` is outside
            [0, 1], or ``subtype`` is invalid for ``type``.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    type: HypothesisType
    subtype: str | None = None
    status: Status
    evidence_strength: float
    severity: Severity | None = None
    priority_rank: int | None = None
    trigger: TriggerResult
    diagnostic_evidence_ids: list[str] = Field(default_factory=list)
    experiment_ids: list[str] = Field(default_factory=list)
    effect: EffectResult | None = None
    decision: Decision
    confounders_checked: list[str] = Field(default_factory=list)
    caveats: list[str] = Field(default_factory=list)
    recommended_actions: list[str] = Field(default_factory=list)
    explanation: ExplanationOutput | None = None

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("HypothesisResult id must not be empty")
        return v

    @field_validator("evidence_strength")
    @classmethod
    def _strength_in_range(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError(f"evidence_strength must be in [0, 1], got {v}")
        return v

    @model_validator(mode="after")
    def _subtype_valid_for_type(self) -> HypothesisResult:
        """Reject subtype values that are not valid for this hypothesis type."""
        allowed = _VALID_SUBTYPES.get(self.type, {None})
        if self.subtype not in allowed:
            raise ValueError(
                f"subtype {self.subtype!r} is not valid for hypothesis type "
                f"{self.type.value!r}. Allowed: {allowed!r}"
            )
        return self
