"""Result contracts: ColumnProfile, DatasetProfile, AssessmentResult, AnalysisResult.

``AnalysisResult`` is the single top-level artifact produced by a complete
analysis run.  It carries ``schema_version = "1.0"`` and a
``reproducibility_hash`` so that identical configs on identical data always
produce the same verdicts.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from modelroot.schemas.common import SCHEMA_VERSION, Algorithm
from modelroot.schemas.config import AnalysisConfig
from modelroot.schemas.evidence import Evidence
from modelroot.schemas.experiment import ExperimentResult
from modelroot.schemas.hypothesis import HypothesisResult
from modelroot.schemas.metrics import MetricResult

# ---------------------------------------------------------------------------
# ColumnProfile
# ---------------------------------------------------------------------------


class ColumnProfile(BaseModel):
    """Profile of a single dataset column.

    Attributes:
        name: Column name.
        inferred_type: Inferred column type — ``"numeric"`` or ``"categorical"``.
        missing_frac: Fraction of rows with a missing value in [0, 1].
        n_unique: Number of distinct non-null values.
        unique_ratio: ``n_unique / n_rows``.
        dropped: Whether the column was dropped before modelling.
        reason: Human-readable reason why the column was dropped (if applicable).
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    inferred_type: str
    missing_frac: float
    n_unique: int
    unique_ratio: float
    dropped: bool = False
    reason: str | None = None


# ---------------------------------------------------------------------------
# DatasetProfile
# ---------------------------------------------------------------------------


class DatasetProfile(BaseModel):
    """Summary statistics for the dataset after loading and validation.

    Attributes:
        n_rows: Number of rows in the dataset.
        n_cols: Number of columns (including the target).
        class_balance: Mapping from class label to row count.
        columns: Per-column profiles.
        warnings: Free-text warnings raised during profiling.
    """

    model_config = ConfigDict(extra="forbid")

    n_rows: int
    n_cols: int
    class_balance: dict[str, int]
    columns: list[ColumnProfile] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ReferenceResult  (nested inside AssessmentResult)
# ---------------------------------------------------------------------------


class ReferenceResult(BaseModel):
    """Performance of the reference panel model on the holdout set.

    Attributes:
        algorithm: Which algorithm was used for the reference model.
        metrics: Metrics achieved by the reference model.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Algorithm
    metrics: list[MetricResult] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AssessmentResult
# ---------------------------------------------------------------------------


class AssessmentResult(BaseModel):
    """Verdict on whether the user's model is underperforming.

    Attributes:
        baseline_metrics: Metrics for the user's model on train / holdout.
        dummy_metrics: Metrics for the majority-class dummy baseline.
        reference: Best reference-panel model result.
        headroom: Gap between reference and baseline on the primary metric.
        train_holdout_gap: ``train_score - holdout_score`` on the primary metric.
        current_drop: ``holdout_score - current_score`` if a labeled current
            dataset was supplied, otherwise ``None``.
        underperforming: Whether the engine concluded the model underperforms.
        reasons: List of human-readable reasons for the verdict.
        headline: One-line summary string shown to the user.
    """

    model_config = ConfigDict(extra="forbid")

    baseline_metrics: list[MetricResult] = Field(default_factory=list)
    dummy_metrics: list[MetricResult] = Field(default_factory=list)
    reference: ReferenceResult
    headroom: float
    train_holdout_gap: float
    current_drop: float | None = None
    underperforming: bool
    reasons: list[str] = Field(default_factory=list)
    headline: str


# ---------------------------------------------------------------------------
# AnalysisResult  (top-level artifact)
# ---------------------------------------------------------------------------

#: Status values allowed on a top-level AnalysisResult.
AnalysisStatus = str  # "completed" | "partial" | "failed"


class AnalysisResult(BaseModel):
    """Complete output of one ModelRoot analysis run.

    This is the single top-level artifact.  It carries ``schema_version``
    for forward-compatibility and a ``reproducibility_hash`` that is stable
    across identical runs.

    Attributes:
        schema_version: Schema version string, always ``"1.0"`` for this
            release.  Additive optional fields bump the minor version;
            removals / renames bump the major version.
        analysis_id: Globally unique run identifier (UUID).
        created_at: ISO-8601 timestamp of when the run started.
        status: Terminal state — ``"completed"``, ``"partial"``, or
            ``"failed"``.
        config: The ``AnalysisConfig`` that produced this result.
        dataset_fingerprint: Hash of the dataset used for reproducibility.
        split_fingerprint: Hash of the train/holdout split.
        versions: Library version strings (e.g. ``{"modelroot": "0.0.1"}``).
        rule_set_version: Version of the decision rule set applied.
        thresholds: Snapshot of the ``Thresholds`` object used.  Stored as a
            generic JSON object in M1; the ``Thresholds`` Pydantic model is
            introduced in M6.
        assessment: Assessment verdict.
        hypotheses: One result per triggered hypothesis (or sub-hypothesis).
        experiments: All experiment results produced during the analysis run.
        evidence: All evidence items cited by hypothesis results.
        warnings: Engine-level warnings (e.g. budget exceeded, partial run).
        timings: Wall-clock timings per pipeline stage.
        reproducibility_hash: Hash of config + dataset + split + thresholds +
            rule_set_version + library versions.  Same hash ⇒ identical
            verdicts (excluding timings).

    Raises:
        ValueError: If ``schema_version`` is empty or ``analysis_id`` is empty.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: str = SCHEMA_VERSION
    analysis_id: str
    created_at: datetime
    status: str  # "completed" | "partial" | "failed"
    config: AnalysisConfig
    dataset_fingerprint: str
    split_fingerprint: str
    versions: dict[str, str] = Field(default_factory=dict)
    rule_set_version: str
    thresholds: dict[str, Any] = Field(default_factory=dict)
    assessment: AssessmentResult
    hypotheses: list[HypothesisResult] = Field(default_factory=list)
    experiments: list[ExperimentResult] = Field(default_factory=list)
    evidence: list[Evidence] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    timings: dict[str, float] = Field(default_factory=dict)
    reproducibility_hash: str

    @field_validator("schema_version")
    @classmethod
    def _schema_version_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("schema_version must not be empty")
        return v

    @field_validator("analysis_id")
    @classmethod
    def _analysis_id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("analysis_id must not be empty")
        return v

    @field_validator("status")
    @classmethod
    def _status_valid(cls, v: str) -> str:
        allowed = {"completed", "partial", "failed"}
        if v not in allowed:
            raise ValueError(f"status must be one of {allowed!r}, got {v!r}")
        return v

    @field_validator("reproducibility_hash")
    @classmethod
    def _hash_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("reproducibility_hash must not be empty")
        return v
