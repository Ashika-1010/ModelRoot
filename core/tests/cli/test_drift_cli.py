"""Tests for ``modelroot drift`` CLI subcommand (M19)."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from modelroot.cli.main import main


def test_drift_cli_help() -> None:
    """Verifies that --help displays options and exits cleanly."""
    with pytest.raises(SystemExit) as exc:
        main(["drift", "--help"])
    assert exc.value.code == 0


def test_drift_cli_execution(tmp_path: Path) -> None:
    """Verifies end-to-end execution of drift subcommand."""
    ref_path = tmp_path / "ref.csv"
    cur_path = tmp_path / "cur.csv"
    out_path = tmp_path / "drift.json"

    df_ref = pd.DataFrame({"f1": [1.0, 2.0, 3.0, 4.0] * 20, "c1": ["a", "b"] * 40})
    df_cur = pd.DataFrame({"f1": [2.0, 3.0, 4.0, 5.0] * 20, "c1": ["a", "b"] * 40})

    df_ref.to_csv(ref_path, index=False)
    df_cur.to_csv(cur_path, index=False)

    with pytest.raises(SystemExit) as exc:
        main(
            [
                "drift",
                "--reference",
                str(ref_path),
                "--current",
                str(cur_path),
                "--n-permutations",
                "5",
                "--out",
                str(out_path),
            ]
        )
    assert exc.value.code == 0

    assert out_path.exists()
    data = json.loads(out_path.read_text(encoding="utf-8"))
    assert "domain_classifier" in data
    assert "univariate_drift" in data
    assert "f1" in data["univariate_drift"]


def test_drift_cli_missing_file(tmp_path: Path) -> None:
    """Verifies error handling when input CSV does not exist."""
    cur_path = tmp_path / "cur.csv"
    pd.DataFrame({"x": [1, 2]}).to_csv(cur_path, index=False)

    with pytest.raises(SystemExit) as exc:
        main(["drift", "--reference", "non_existent.csv", "--current", str(cur_path)])
    assert exc.value.code == 2
