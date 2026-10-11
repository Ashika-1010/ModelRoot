"""Verify that all public models, enums, and constants are importable
from ``modelroot.schemas`` at the top level.

This test acts as a canary: if any re-export is accidentally removed from
``schemas/__init__.py``, exactly one test here will fail with a clear name.
"""

from __future__ import annotations

import pytest


@pytest.mark.parametrize(
    "name",
    [
        # Constant
        "SCHEMA_VERSION",
        # Enums
        "Algorithm",
        "EvidenceKind",
        "ExperimentStatus",
        "HypothesisType",
        "PrimaryMetric",
        "RulesProfile",
        "Severity",
        "SplitMode",
        "Status",
        # Config models
        "AnalysisConfig",
        "JsonScalar",
        "Metadata",
        "ModelSpec",
        "SplitConfig",
        # Metrics
        "MetricResult",
        # Evidence
        "Evidence",
        # Experiment
        "ExperimentResult",
        "ExperimentSpec",
        "NBResult",
        # Hypothesis
        "Decision",
        "EffectResult",
        "ExplanationOutput",
        "HypothesisResult",
        "TriggerResult",
        # Result
        "AnalysisResult",
        "AssessmentResult",
        "ColumnProfile",
        "DatasetProfile",
        "ReferenceResult",
        # Export helpers
        "export_json_schemas",
        "get_export_registry",
    ],
)
def test_public_name_importable(name: str) -> None:
    """Every public name must be importable directly from ``modelroot.schemas``."""
    import modelroot.schemas as schemas

    assert hasattr(schemas, name), (
        f"{name!r} is not exported from modelroot.schemas. "
        f"Add it to schemas/__init__.py __all__ and re-export."
    )


def test_schema_version_value() -> None:
    from modelroot.schemas import SCHEMA_VERSION

    assert SCHEMA_VERSION == "1.0"


def test_export_json_schemas_callable() -> None:
    from modelroot.schemas import export_json_schemas

    assert callable(export_json_schemas)


def test_get_export_registry_returns_nonempty_list() -> None:
    from modelroot.schemas import get_export_registry

    registry = get_export_registry()
    assert isinstance(registry, list)
    assert len(registry) >= 1
    # Each entry is a (name, class) pair
    for name, cls in registry:
        assert isinstance(name, str) and name
        assert isinstance(cls, type)
