"""Smoke tests for the modelroot package (M0).

Test 1 — import + version:
    Verifies that ``import modelroot`` succeeds and that
    ``modelroot.__version__`` equals ``"0.0.1"``.

Test 2 — strict-marker enforcement:
    Verifies that pytest's ``--strict-markers`` flag causes an *error* (not
    just a warning) when a test is decorated with an unregistered marker.

    Strategy: we invoke pytest as a subprocess against a tiny temporary test
    file that uses an unknown marker.  We assert the subprocess exits with a
    non-zero code, which is the only reliable way to confirm strict-marker
    rejection without making *this* test file itself fail.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import textwrap

# ---------------------------------------------------------------------------
# Test 1 — import and version
# ---------------------------------------------------------------------------


def test_import_modelroot() -> None:
    """Importing ``modelroot`` must succeed."""
    import modelroot

    assert modelroot is not None


def test_version() -> None:
    """``modelroot.__version__`` must equal ``"0.0.1"``."""
    import modelroot

    assert modelroot.__version__ == "0.0.1"


# ---------------------------------------------------------------------------
# Test 2 — strict-marker enforcement
# ---------------------------------------------------------------------------


def test_strict_markers_rejects_unknown_marker(tmp_path: pathlib.Path) -> None:
    """Pytest must *fail* when a test uses an unregistered marker.

    This test spawns a subprocess that runs pytest with ``--strict-markers``
    on a tiny synthetic test file that decorates a test with the marker
    ``unregistered_marker_xyz``, which is deliberately absent from
    ``pyproject.toml``.  The expected outcome is a non-zero exit code from
    pytest (exit code 4 = usage error / collection error with strict
    markers).

    We do NOT just inspect configuration strings — we require the actual
    pytest process to exit non-zero, which is the behavioural guarantee.

    Args:
        tmp_path: pytest built-in fixture providing a temporary directory.

    Raises:
        AssertionError: if pytest exits with code 0 (strict-marker check
            would be broken).
    """
    bad_test = tmp_path / "test_bad_marker.py"
    bad_test.write_text(
        textwrap.dedent(
            """\
            import pytest

            @pytest.mark.unregistered_marker_xyz
            def test_dummy():
                pass
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "--strict-markers",
            "--no-header",
            "-q",
            str(bad_test),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, (
        "Expected pytest to exit non-zero when an unregistered marker is used "
        f"with --strict-markers, but it exited with code {result.returncode}.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )

    # Confirm the failure is specifically about the unknown marker, not some
    # unrelated error.
    combined = result.stdout + result.stderr
    assert "unregistered_marker_xyz" in combined, (
        "Expected the marker name to appear in pytest output, but it did not.\n"
        f"stdout:\n{result.stdout}\n"
        f"stderr:\n{result.stderr}"
    )
