"""Experiment contracts: ExperimentSpec, NBResult, ExperimentResult.

An *experiment* compares the baseline model against a single intervention
on a defined evaluation set and produces a paired bootstrap result.
"""

from __future__ import annotations

from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelroot.schemas.common import ExperimentStatus
from modelroot.schemas.config import JsonScalar, _validate_json_scalar_dict

# ---------------------------------------------------------------------------
# ExperimentSpec
# ---------------------------------------------------------------------------


class ExperimentSpec(BaseModel):
    """Specification describing one experiment to run.

    Attributes:
        id: Unique, non-empty experiment identifier.
        hypothesis_id: ID of the hypothesis that generated this experiment.
        name: Human-readable name.
        intervention: Registered intervention name (e.g. ``"regularize"``).
        params: Intervention parameters.  Values must be JSON scalars.
        metric: Name of the metric to use for comparison.
        eval_on: Which evaluation sets to score
            (``"holdout"``, ``"current"``, or both).
        n_boot: Number of bootstrap resamples.
        repeats: Number of repeated-subsample repeats.
        seed: Random seed for this experiment.  Must be ≥ 0.
        budget_s: Per-experiment wall-clock budget in seconds.  Must be > 0.

    Raises:
        ValueError: If ``id`` or ``hypothesis_id`` are empty, ``seed < 0``,
            ``budget_s <= 0``, or any param value is a non-scalar.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    hypothesis_id: str
    name: str
    intervention: str
    params: Annotated[dict[str, JsonScalar], Field(default_factory=dict)]
    metric: str
    eval_on: list[str] = Field(default_factory=lambda: ["holdout"])
    n_boot: int = 1000
    repeats: int = 10
    seed: int
    budget_s: float

    @field_validator("id")
    @classmethod
    def _id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("ExperimentSpec id must not be empty")
        return v

    @field_validator("hypothesis_id")
    @classmethod
    def _hypothesis_id_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("hypothesis_id must not be empty")
        return v

    @field_validator("seed")
    @classmethod
    def _seed_non_negative(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"seed must be >= 0, got {v}")
        return v

    @field_validator("budget_s")
    @classmethod
    def _budget_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"budget_s must be > 0, got {v}")
        return v

    @field_validator("params", mode="before")
    @classmethod
    def _params_scalars(cls, v: Any) -> Any:
        if isinstance(v, dict):
            _validate_json_scalar_dict(v)
        return v


# ---------------------------------------------------------------------------
# NBResult - Nadeau-Bengio corrected resampled t-test output
# ---------------------------------------------------------------------------


class NBResult(BaseModel):
    """Output of the Nadeau-Bengio corrected resampled t-test.

    Attributes:
        mean: Mean delta across repeated subsamples.
        se: Corrected standard error.
        t: t-statistic.
        df: Degrees of freedom (J - 1 where J = number of repeats).
        p: Two-sided p-value.
        ci_low: Lower bound of the corrected confidence interval.
        ci_high: Upper bound of the corrected confidence interval.

    Raises:
        ValueError: If ``ci_low > ci_high``.
    """

    model_config = ConfigDict(extra="forbid")

    mean: float
    se: float
    t: float
    df: float
    p: float
    ci_low: float
    ci_high: float

    @model_validator(mode="after")
    def _ci_ordering(self) -> NBResult:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")
        return self


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


class ExperimentResult(BaseModel):
    """Result of a completed (or attempted) experiment.

    Attributes:
        spec_id: References ``ExperimentSpec.id``.
        status: Terminal state of the experiment run.
        baseline_metric: Baseline model metric value on the evaluation set.
        intervention_metric: Intervention model metric value.
        delta: ``intervention_metric - baseline_metric``.
        ci_low: Lower bound of the paired bootstrap CI on delta.
        ci_high: Upper bound of the paired bootstrap CI on delta.
        n_eval: Number of evaluation-set samples.
        n_valid_boot: Number of valid bootstrap resamples (may be < n_boot
            if some resamples contained only one class).
        repeat_deltas: Delta values from repeated-subsample runs.
        consistency: Fraction of repeats with delta > 0.
        nb: Nadeau-Bengio corrected test output, if repeats were run.
        runtime_s: Wall-clock time in seconds.
        notes: Free-text notes recorded during the run.
        error: Error message if ``status`` is ``error``.

    Raises:
        ValueError: If ``ci_low > ci_high``.
    """

    model_config = ConfigDict(extra="forbid")

    spec_id: str
    status: ExperimentStatus
    baseline_metric: float
    intervention_metric: float
    delta: float
    ci_low: float
    ci_high: float
    n_eval: int
    n_valid_boot: int
    repeat_deltas: list[float] = Field(default_factory=list)
    consistency: float | None = None
    nb: NBResult | None = None
    runtime_s: float
    notes: list[str] = Field(default_factory=list)
    error: str | None = None

    @model_validator(mode="after")
    def _ci_ordering(self) -> ExperimentResult:
        if self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")
        return self
