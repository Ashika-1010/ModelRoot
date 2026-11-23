"""Benchmark contracts and scenario models (M12).

Defines the core data contracts for the ModelRoot failure injection benchmark:
- :class:`ScenarioSpec`: specification of a benchmark scenario.
- :class:`GroundTruth`: ground-truth failure annotations and impact.
- :class:`ScenarioResult`: summary of diagnosis engine execution on a scenario.
- :class:`BenchmarkRecord`: container pairing spec, ground truth, and result.
"""

from __future__ import annotations

from enum import Enum
from typing import Any, Literal

from modelroot.schemas.config import ModelSpec
from pydantic import BaseModel, ConfigDict, Field

ScenarioClass = Literal[
    "clean_control",
    "single_failure",
    "multi_failure",
    "distractor",
    "hard_negative",
]

Role = Literal["dev", "heldout"]


class ScenarioClassEnum(str, Enum):
    """Enumeration of scenario classes."""

    clean_control = "clean_control"
    single_failure = "single_failure"
    multi_failure = "multi_failure"
    distractor = "distractor"
    hard_negative = "hard_negative"


class RoleEnum(str, Enum):
    """Dataset and scenario evaluation roles."""

    dev = "dev"
    heldout = "heldout"


class InjectionSpec(BaseModel):
    """Specification of an injected failure or distractor.

    Attributes:
        label: Failure identifier (e.g., 'overfitting', 'imbalance', 'leakage:duplicate').
        severity: Severity designation (e.g., 'S1', 'S2', 'S3').
        params: Free-form parameter dictionary for the injector.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    severity: str = ""
    params: dict[str, Any] = Field(default_factory=dict)
    expected_subtypes: dict[str, bool] = Field(default_factory=dict)


class ScenarioSpec(BaseModel):
    """Full specification of a benchmark scenario.

    Attributes:
        id: Unique, deterministic scenario identifier.
        dataset: Base dataset identifier.
        model_spec: Baseline estimator algorithm and hyperparameters.
        scenario_class: Scenario classification (e.g. single_failure, clean_control).
        injections: Ordered list of failure injections applied.
        seed: Random seed for data generation and model fitting.
        role: Scenario role (dev for threshold tuning; heldout for evaluation).
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    dataset: str
    model_spec: ModelSpec | dict[str, Any]
    scenario_class: ScenarioClass
    injections: list[InjectionSpec] = Field(default_factory=list)
    seed: int = 0
    role: Role = "dev"


class InjectedFailure(BaseModel):
    """A ground-truth record of an injected failure.

    Attributes:
        label: Standard hypothesis or sub-claim label.
        severity: Injected severity string.
    """

    model_config = ConfigDict(extra="forbid")

    label: str
    severity: str = ""


class GroundTruth(BaseModel):
    """Ground-truth annotations for a scenario.

    Attributes:
        injected: List of truly injected failures.
        distractors: List of distractor perturbations applied.
        impact: Performance degradation (clean - scenario metric) on oracle holdout.
        no_effect: True if impact < 0.02 (excluded from recall denominator).
    """

    model_config = ConfigDict(extra="forbid")

    injected: list[InjectedFailure] = Field(default_factory=list)
    distractors: list[str] = Field(default_factory=list)
    impact: float = 0.0
    no_effect: bool = False
    expected_subtypes: dict[str, bool] = Field(default_factory=dict)


class ScenarioResult(BaseModel):
    """Result of running the ModelRoot diagnosis engine on a scenario.

    Attributes:
        scenario_id: Identifier matching ScenarioSpec.id.
        statuses: Mapping from hypothesis label to diagnosed status string.
        top1: Label of the top-ranked supported diagnosis, or None.
        runtime_s: Wall-clock execution time in seconds.
        analysis_hash: Reproducibility hash from AnalysisResult.
        control_statuses: Statuses from matching clean control (for adjusted FP).
        effects: Additive mapping of hypothesis label to effect delta (for MRR).
        ranking: Additive ordered list of hypothesis labels from top to bottom.
    """

    model_config = ConfigDict(extra="forbid")

    scenario_id: str
    statuses: dict[str, str] = Field(default_factory=dict)
    top1: str | None = None
    runtime_s: float = 0.0
    analysis_hash: str = ""
    control_statuses: dict[str, str] = Field(default_factory=dict)
    effects: dict[str, float] = Field(default_factory=dict)
    ranking: list[str] = Field(default_factory=list)


class BenchmarkRecord(BaseModel):
    """Container pairing a scenario specification, ground truth, and optional result.

    Attributes:
        spec: Scenario specification.
        ground_truth: Ground-truth annotations.
        result: Execution result (populated once run).
    """

    model_config = ConfigDict(extra="forbid")

    spec: ScenarioSpec
    ground_truth: GroundTruth
    result: ScenarioResult | None = None
