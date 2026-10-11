"""JSON Schema export for all public ModelRoot contract models.

Usage — regenerate snapshots from the repository root::

    python -c "
    from modelroot.schemas.export import export_json_schemas
    export_json_schemas('docs/schemas')
    "

Or run the snapshot test, which re-exports and diffs against committed files::

    pytest core/tests/schemas/test_snapshot.py

The exported files are deterministic (no timestamps, stable key ordering)
so they can be committed and diff'd reliably.
"""

from __future__ import annotations

import json
import pathlib

from pydantic import BaseModel

from modelroot.schemas.common import SCHEMA_VERSION
from modelroot.schemas.config import (
    AnalysisConfig,
    Metadata,
    ModelSpec,
    SplitConfig,
)
from modelroot.schemas.evidence import Evidence
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec, NBResult
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

# ---------------------------------------------------------------------------
# Registry of public contract models to export
# ---------------------------------------------------------------------------

#: Ordered mapping from file-stem to Pydantic model class.
#: Only top-level public contracts are exported; nested helper models
#: (e.g. AblationConfig, which is only accessible via AnalysisConfig.ablation)
#: are reachable via $defs in the schemas that use them and are not exported
#: as standalone files.
_EXPORT_REGISTRY: list[tuple[str, type[BaseModel]]] = [
    ("ModelSpec", ModelSpec),
    ("SplitConfig", SplitConfig),
    ("Metadata", Metadata),
    ("AnalysisConfig", AnalysisConfig),
    ("MetricResult", MetricResult),
    ("Evidence", Evidence),
    ("ExperimentSpec", ExperimentSpec),
    ("NBResult", NBResult),
    ("ExperimentResult", ExperimentResult),
    ("TriggerResult", TriggerResult),
    ("EffectResult", EffectResult),
    ("ExplanationOutput", ExplanationOutput),
    ("Decision", Decision),
    ("HypothesisResult", HypothesisResult),
    ("ColumnProfile", ColumnProfile),
    ("DatasetProfile", DatasetProfile),
    ("ReferenceResult", ReferenceResult),
    ("AssessmentResult", AssessmentResult),
    ("AnalysisResult", AnalysisResult),
]


def export_json_schemas(directory: str | pathlib.Path) -> None:
    """Write JSON Schema files for all public contract models to *directory*.

    Each model is written to ``<directory>/<ModelName>.json``.  Files are
    created deterministically: keys are sorted, indentation is 2 spaces, and
    no timestamps or volatile values are included.

    The function creates *directory* (and any parents) if it does not exist.
    Existing files are overwritten.

    Args:
        directory: Target directory path (string or ``pathlib.Path``).

    Raises:
        OSError: If the directory cannot be created or files cannot be written.
    """
    out_dir = pathlib.Path(directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    for name, model_cls in _EXPORT_REGISTRY:
        schema = model_cls.model_json_schema()
        # Inject schema_version into top-level artifact schemas
        if name == "AnalysisResult":
            schema.setdefault("x-schema-version", SCHEMA_VERSION)
        _write_stable_json(out_dir / f"{name}.json", schema)


def _write_stable_json(path: pathlib.Path, data: object) -> None:
    """Serialise *data* to *path* with stable (sorted-key) formatting.

    Args:
        path: Destination file path.
        data: JSON-serialisable object.
    """
    path.write_text(
        json.dumps(data, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def get_export_registry() -> list[tuple[str, type[BaseModel]]]:
    """Return the ordered list of (name, model_class) pairs that are exported.

    Returns:
        A copy of the export registry list.
    """
    return list(_EXPORT_REGISTRY)
