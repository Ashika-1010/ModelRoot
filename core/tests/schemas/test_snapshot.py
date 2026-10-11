"""JSON Schema snapshot tests.

Strategy
--------
The committed ``docs/schemas/*.json`` files are the source of truth.  These
tests:

1. Re-run ``export_json_schemas()`` into a temporary directory.
2. Load each committed file and the freshly generated file side-by-side.
3. Assert they are identical (same keys, same values, same ordering).

Any change to a model that alters its JSON Schema will cause these tests to
fail.  To **intentionally** update the snapshots:

    python -c "
    from modelroot.schemas.export import export_json_schemas
    export_json_schemas('docs/schemas')
    "

Then commit the updated JSON files alongside the code change.

The test also verifies that the hand-written ``AnalysisResult`` example in
``docs/schemas/examples/analysis_result.json`` still validates against the
live ``AnalysisResult`` model.
"""

from __future__ import annotations

import json
import pathlib

import pytest

from modelroot.schemas import AnalysisResult
from modelroot.schemas.export import export_json_schemas, get_export_registry

# Path to the committed snapshot directory (relative to the repo root).
# We resolve it relative to this test file's location so it works regardless
# of the working directory.
_REPO_ROOT = pathlib.Path(__file__).parent.parent.parent.parent  # core/tests/schemas → repo root
_COMMITTED_DIR = _REPO_ROOT / "docs" / "schemas"
_EXAMPLE_FILE = _COMMITTED_DIR / "examples" / "analysis_result.json"


# ---------------------------------------------------------------------------
# Snapshot equality
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def fresh_schema_dir(tmp_path_factory: pytest.TempPathFactory) -> pathlib.Path:
    """Generate fresh schemas into a temporary directory once per module."""
    out = tmp_path_factory.mktemp("schemas_fresh")
    export_json_schemas(out)
    return out


def _schema_names() -> list[str]:
    """Return the list of model names from the export registry."""
    return [name for name, _ in get_export_registry()]


@pytest.mark.parametrize("model_name", _schema_names())
def test_schema_snapshot_matches_committed(model_name: str, fresh_schema_dir: pathlib.Path) -> None:
    """Freshly generated JSON Schema for *model_name* must match the committed file.

    Args:
        model_name: Name of the model whose schema is being checked.
        fresh_schema_dir: Fixture providing the temporary directory with
            freshly generated schema files.

    Raises:
        AssertionError: If the committed file is missing or its content
            differs from the fresh export.
    """
    committed_file = _COMMITTED_DIR / f"{model_name}.json"
    fresh_file = fresh_schema_dir / f"{model_name}.json"

    assert committed_file.exists(), (
        f"Committed schema file missing: {committed_file}\n"
        f'Regenerate with: python -c "from modelroot.schemas.export import '
        f"export_json_schemas; export_json_schemas('docs/schemas')\""
    )

    committed = json.loads(committed_file.read_text(encoding="utf-8"))
    fresh = json.loads(fresh_file.read_text(encoding="utf-8"))

    assert committed == fresh, (
        f"JSON Schema for {model_name!r} has drifted from the committed snapshot.\n"
        f'To update: python -c "from modelroot.schemas.export import '
        f"export_json_schemas; export_json_schemas('docs/schemas')\"\n"
        f"Then commit the updated docs/schemas/{model_name}.json."
    )


# ---------------------------------------------------------------------------
# Hand-written example validation
# ---------------------------------------------------------------------------


def test_analysis_result_example_validates() -> None:
    """The hand-written AnalysisResult example must pass model validation.

    This ensures the example stays in sync with the schema as the model
    evolves.
    """
    assert _EXAMPLE_FILE.exists(), f"Example file not found: {_EXAMPLE_FILE}"
    data = json.loads(_EXAMPLE_FILE.read_text(encoding="utf-8"))
    result = AnalysisResult.model_validate(data)
    assert result.schema_version == "1.0"
    assert result.analysis_id == "a1b2c3d4-e5f6-7890-abcd-ef1234567890"
    assert result.status == "completed"


# ---------------------------------------------------------------------------
# All registered model names have committed snapshots
# ---------------------------------------------------------------------------


def test_all_registered_models_have_committed_snapshots() -> None:
    """Every model in the export registry must have a corresponding committed file."""
    missing = []
    for name in _schema_names():
        if not (_COMMITTED_DIR / f"{name}.json").exists():
            missing.append(name)
    assert not missing, (
        f"Missing committed snapshots for: {missing}\n"
        f'Run: python -c "from modelroot.schemas.export import '
        f"export_json_schemas; export_json_schemas('docs/schemas')\""
    )
