"""Metric result contract.

A ``MetricResult`` carries a single scalar measurement (optionally with a
bootstrap CI) for one metric on one data split.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, field_validator, model_validator

# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------

#: Valid split labels for metric measurements.
SplitLabel = Literal["train", "holdout", "current", "oof"]


class MetricResult(BaseModel):
    """A scalar metric measurement on a single data split.

    Attributes:
        name: Metric identifier (e.g. ``"macro_f1"``).
        value: Measured value, typically in [0, 1] for classification metrics.
        split: Data split the metric was computed on.
        n: Number of samples used for this measurement.
        ci_low: Lower bound of the bootstrap confidence interval.
        ci_high: Upper bound of the bootstrap confidence interval.

    Raises:
        ValueError: If ``ci_low > ci_high`` when both bounds are supplied.
    """

    model_config = ConfigDict(extra="forbid")

    name: str
    value: float
    split: SplitLabel
    n: int
    ci_low: float | None = None
    ci_high: float | None = None

    @field_validator("name")
    @classmethod
    def _name_non_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("name must not be empty")
        return v

    @model_validator(mode="after")
    def _ci_ordering(self) -> MetricResult:
        """Validate that ci_low <= ci_high when both are supplied."""
        if self.ci_low is not None and self.ci_high is not None and self.ci_low > self.ci_high:
            raise ValueError(f"ci_low ({self.ci_low}) must be <= ci_high ({self.ci_high})")
        return self
