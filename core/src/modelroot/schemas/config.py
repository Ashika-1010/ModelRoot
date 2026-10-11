"""Configuration contracts: ModelSpec, SplitConfig, Metadata, AnalysisConfig.

These models define *what* the user asks ModelRoot to analyse.  They are
validated strictly at ingestion time; every unknown field is rejected.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from modelroot.schemas.common import Algorithm, PrimaryMetric, RulesProfile, SplitMode

# ---------------------------------------------------------------------------
# JSON-scalar type alias used for params dicts
# ---------------------------------------------------------------------------

#: A single JSON scalar value: int, float, str, bool, or None.
#: Lists, dicts, and other complex types are explicitly excluded.
JsonScalar = int | float | str | bool | None


def _validate_json_scalar_dict(value: dict[str, Any]) -> dict[str, Any]:
    """Raise ValueError if any dict value is not a JSON scalar.

    Args:
        value: The params mapping to validate.

    Returns:
        The unchanged mapping if all values are JSON scalars.

    Raises:
        ValueError: If any value is a non-scalar (list, dict, object, …).
    """
    for k, v in value.items():
        if not isinstance(v, int | float | str | bool | type(None)):
            raise ValueError(
                f"params[{k!r}] must be a JSON scalar (int, float, str, bool, or null),"
                f" got {type(v).__name__!r}"
            )
    return value


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


class ModelSpec(BaseModel):
    """Specification of the algorithm and hyper-parameters ModelRoot will train.

    Attributes:
        algorithm: Which algorithm to use.
        params: Algorithm hyper-parameters.  Values must be JSON scalars
            (int, float, str, bool, or null).  Complex types are rejected.
    """

    model_config = ConfigDict(extra="forbid")

    algorithm: Algorithm
    params: Annotated[dict[str, JsonScalar], Field(default_factory=dict)]

    @field_validator("params", mode="before")
    @classmethod
    def _params_must_be_scalars(cls, v: Any) -> Any:
        """Validate that every param value is a JSON scalar."""
        if isinstance(v, dict):
            _validate_json_scalar_dict(v)
        return v


# ---------------------------------------------------------------------------
# SplitConfig
# ---------------------------------------------------------------------------


class SplitConfig(BaseModel):
    """How the dataset is divided into training and test sets.

    Attributes:
        mode: Split strategy.
        test_size: Fraction of rows reserved for the holdout set when
            ``mode`` is ``single_file``.  Must be in (0, 1).
        split_column: Column name whose values are ``"train"`` / ``"test"``
            (required when ``mode`` is ``split_column``).
        test_path: Path to a separate test-set file
            (required when ``mode`` is ``test_file``).
    """

    model_config = ConfigDict(extra="forbid")

    mode: SplitMode
    test_size: float = 0.2
    split_column: str | None = None
    test_path: Path | None = None

    @field_validator("test_size")
    @classmethod
    def _test_size_in_range(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"test_size must be in (0, 1), got {v}")
        return v

    @model_validator(mode="after")
    def _cross_field_consistency(self) -> SplitConfig:
        if self.mode == SplitMode.split_column and not self.split_column:
            raise ValueError("split_column is required when mode is 'split_column'")
        if self.mode == SplitMode.test_file and self.test_path is None:
            raise ValueError("test_path is required when mode is 'test_file'")
        return self


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


class Metadata(BaseModel):
    """Optional contextual hints about the dataset and task.

    Attributes:
        expected_score: The user's prior expectation of model performance on
            the primary metric.  Used by the assessor to identify
            underperformance.
        unavailable_at_inference: Column names that exist in training data
            but are *not* available at inference time (potential leakage).
        id_columns: Columns that are row identifiers (excluded from features).
        sensitive_columns: Columns corresponding to protected attributes
            (used by the subgroup hypothesis).
    """

    model_config = ConfigDict(extra="forbid")

    expected_score: float | None = None
    unavailable_at_inference: list[str] = Field(default_factory=list)
    id_columns: list[str] = Field(default_factory=list)
    sensitive_columns: list[str] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


class AblationConfig(BaseModel):
    """Flags controlling ablation modes used by the benchmark.

    Attributes:
        subgroup_protocol: Whether to apply the full two-stage FDR subgroup
            protocol (``full``) or the naive single-pass ranking ablation
            (``naive``).
    """

    model_config = ConfigDict(extra="forbid")

    subgroup_protocol: RulesProfile = RulesProfile.full


# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------


class AnalysisConfig(BaseModel):
    """Top-level configuration for a single ModelRoot analysis run.

    This is the primary user-facing input object.  It must be fully
    specified before an analysis can start.

    Attributes:
        data_path: Path to the primary CSV dataset.
        target: Name of the target column.
        positive_class: Value in ``target`` that represents the positive class.
        split: How the dataset is split into train / holdout sets.
        current_path: Optional path to a third "current" dataset collected
            after deployment (used by H5 — distribution shift).
        model_spec: Algorithm and hyper-parameters to train.
        primary_metric: Performance metric used as the primary signal.
        metadata: Optional contextual hints.
        seed: Master random seed.  Must be ≥ 0.
        budget_s: Wall-clock budget in seconds for the experiment phase.
            Must be > 0.
        run_experiments: When ``False`` the engine only runs triggers
            (detector-only ablation mode, never shown to end users).
        rules_profile: Decision-rule strictness profile.
        ablation: Fine-grained ablation flags.
        explain: Whether to generate natural-language explanations.
    """

    model_config = ConfigDict(extra="forbid")

    data_path: Path
    target: str
    positive_class: str
    split: SplitConfig = Field(default_factory=lambda: SplitConfig(mode=SplitMode.single_file))
    current_path: Path | None = None
    model_spec: ModelSpec
    primary_metric: PrimaryMetric = PrimaryMetric.macro_f1
    metadata: Metadata = Field(default_factory=Metadata)
    seed: int = 0
    budget_s: float = 300.0
    run_experiments: bool = True
    rules_profile: RulesProfile = RulesProfile.full
    ablation: AblationConfig = Field(default_factory=AblationConfig)
    explain: bool = True

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
