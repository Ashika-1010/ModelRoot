"""ModelRoot schemas package — all public data contracts.

Every model, enum, and constant in this package is a stable API contract.
Import them directly from ``modelroot.schemas``::

    from modelroot.schemas import (
        SCHEMA_VERSION,
        Algorithm,
        AnalysisConfig,
        AnalysisResult,
        # …
    )

To regenerate the committed JSON Schema snapshots, run::

    python -c "
    from modelroot.schemas import export_json_schemas
    export_json_schemas('docs/schemas')
    "
"""

from __future__ import annotations

# Re-export everything from sub-modules so users never need to know the
# internal file structure.
from modelroot.schemas.common import (
    SCHEMA_VERSION,
    Algorithm,
    EvidenceKind,
    ExperimentStatus,
    HypothesisType,
    PrimaryMetric,
    RulesProfile,
    Severity,
    SplitMode,
    Status,
)
from modelroot.schemas.config import (
    AnalysisConfig,
    JsonScalar,
    Metadata,
    ModelSpec,
    SplitConfig,
)
from modelroot.schemas.evidence import Evidence
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec, NBResult
from modelroot.schemas.export import export_json_schemas, get_export_registry
from modelroot.schemas.hypothesis import (
    Decision,
    EffectResult,
    ExplanationOutput,
    HypothesisResult,
    TriggerResult,
)
from modelroot.schemas.metrics import MetricResult
from modelroot.schemas.result import (
    AnalysisResult,
    AssessmentResult,
    ColumnProfile,
    DatasetProfile,
    ReferenceResult,
)

__all__ = [
    "SCHEMA_VERSION",
    "Algorithm",
    "AnalysisConfig",
    "AnalysisResult",
    "AssessmentResult",
    "ColumnProfile",
    "DatasetProfile",
    "Decision",
    "EffectResult",
    "Evidence",
    "EvidenceKind",
    "ExperimentResult",
    "ExperimentSpec",
    "ExperimentStatus",
    "ExplanationOutput",
    "HypothesisResult",
    "HypothesisType",
    "JsonScalar",
    "Metadata",
    "MetricResult",
    "ModelSpec",
    "NBResult",
    "PrimaryMetric",
    "ReferenceResult",
    "RulesProfile",
    "Severity",
    "SplitConfig",
    "SplitMode",
    "Status",
    "TriggerResult",
    "export_json_schemas",
    "get_export_registry",
]
