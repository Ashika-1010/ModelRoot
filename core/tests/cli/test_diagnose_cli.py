"""M11 CLI tests: T12-T13."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import pytest

# Ensure regularize intervention is registered
import modelroot.hypotheses.overfitting  # noqa: F401


def _make_overfit_csv(
    tmp_path: pathlib.Path, n_train: int = 150, n_test: int = 300
) -> pathlib.Path:
    """Write a split-column CSV to tmp_path and return its path."""
    from fixtures.synthetic import make_overfit

    return make_overfit(tmp_path, n_train=n_train, n_test=n_test, n_noise=10, seed=42)


# ---------------------------------------------------------------------------
# T12: test_cli_diagnose (in-process)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cli_diagnose(tmp_path: pathlib.Path) -> None:
    """Happy path, JSON output, no-experiments, naive profile, exit codes."""
    from modelroot.cli.main import main

    csv = _make_overfit_csv(tmp_path)
    out = tmp_path / "result.json"

    # Happy path
    with pytest.raises(SystemExit) as exc_info:
        main(
            [
                "diagnose",
                str(csv),
                "--target",
                "label",
                "--positive-class",
                "pos",
                "--algorithm",
                "decision_tree",
                "--metric",
                "macro_f1",
                "--seed",
                "0",
                "--out",
                str(out),
            ]
        )
    assert exc_info.value.code == 0
    assert out.exists()
    data = json.loads(out.read_text())
    assert "hypotheses" in data
    assert data["status"] == "completed"

    # No-experiments mode
    out_noexp = tmp_path / "noexp.json"
    with pytest.raises(SystemExit) as exc_info2:
        main(
            [
                "diagnose",
                str(csv),
                "--target",
                "label",
                "--positive-class",
                "pos",
                "--algorithm",
                "decision_tree",
                "--no-experiments",
                "--out",
                str(out_noexp),
            ]
        )
    assert exc_info2.value.code == 0
    data_noexp = json.loads(out_noexp.read_text())
    assert data_noexp["experiments"] == []

    # Naive profile
    out_naive = tmp_path / "naive.json"
    with pytest.raises(SystemExit) as exc_info3:
        main(
            [
                "diagnose",
                str(csv),
                "--target",
                "label",
                "--positive-class",
                "pos",
                "--algorithm",
                "decision_tree",
                "--rules-profile",
                "naive",
                "--out",
                str(out_naive),
            ]
        )
    assert exc_info3.value.code == 0

    # Exit code 2 on bad target
    with pytest.raises(SystemExit) as exc_info4:
        main(["diagnose", str(csv), "--target", "nonexistent_column"])
    assert exc_info4.value.code == 2

    # Exit code 2 on invalid JSON params
    with pytest.raises(SystemExit) as exc_info5:
        main(["diagnose", str(csv), "--target", "label", "--params", "{bad json}"])
    assert exc_info5.value.code == 2


# ---------------------------------------------------------------------------
# T13: test_cli_diagnose_subprocess_smoke
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_cli_diagnose_subprocess_smoke(tmp_path: pathlib.Path) -> None:
    """One real subprocess execution of modelroot diagnose."""
    csv = _make_overfit_csv(tmp_path)
    out = tmp_path / "smoke.json"

    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modelroot",
            "diagnose",
            str(csv),
            "--target",
            "label",
            "--positive-class",
            "pos",
            "--algorithm",
            "decision_tree",
            "--seed",
            "0",
            "--out",
            str(out),
        ],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, f"stderr: {result.stderr[:500]}"
    assert out.exists()
    data = json.loads(out.read_text())
    assert data["status"] == "completed"
    assert "hypotheses" in data
    # No tracebacks in stdout
    assert "Traceback" not in result.stdout
