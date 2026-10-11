"""Evidence contract.

An ``Evidence`` item is an atomic, ID-tagged measured fact.  Explanations
may only cite evidence IDs that appear in the analysis result's ``evidence``
list.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from modelroot.schemas.common import EvidenceKind

# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


class Evidence(BaseModel):
    """A single, atomic measured fact produced during an analysis.

    Evidence items are the only things explanations may cite.  Each item
    carries a stable ``id`` computed from its content (not from position).

    Attributes:
        id: Stable identifier: ``"ev_"`` followed by 8 hex characters derived
            from a hash of ``(name, value, method, experiment_id)``.
        kind: Whether this is a passive diagnostic measurement or the result
            of a paired experiment.
        name: Human-readable name for the measured quantity.
        value: Numeric value of the measurement.
        ci_low: Lower bound of the bootstrap confidence interval.
        ci_high: Upper bound of the bootstrap confidence interval.
        n: Number of samples underlying the measurement.
        method: Name of the statistical method or computation used.
        hint: Short, template-friendly description of what this evidence means.
        experiment_id: ``ExperimentSpec.id`` if this evidence originated from
            an experiment; ``None`` for diagnostic evidence.

    Raises:
        ValueError: If ``id`` is empty, or ``ci_low > ci_high`` when both are
            supplied.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    kind: EvidenceKind
    name: str
    value: float
    ci_low: float | None = None
    ci_high: float | None = None
    n: int | None = None
    method: str
    hint: str
    experiment_id: str | None = None

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Evidence id must not be empty")
        return v

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Evidence name must not be empty")
        return v

    @field_validator("method")
    @classmethod
    def _method_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Evidence method must not be empty")
        return v

    @model_validator(mode="after")
    def _ci_ordering(self) -> Evidence:
        """Validate that ci_low <= ci_high when both are supplied."""
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")
        return self
