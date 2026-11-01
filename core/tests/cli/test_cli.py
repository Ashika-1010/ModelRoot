"""Tests for M6 CLI (T6-15 to T6-19)."""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

import numpy as np
import pandas as pd
import pytest


def _make_csv(tmp_path: pathlib.Path, n: int = 80) -> pathlib.Path:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 70, size=n).astype(str),
            "income": rng.integers(30000, 120000, size=n).astype(str),
            "label": ["yes" if rng.random() > 0.5 else "no" for _ in range(n)],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    return p


# T6-15: --help exits 0
def test_cli_help_exit_zero() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "modelroot", "assess", "--help"], capture_output=True, text=True
    )
    assert result.returncode == 0
    assert "assess" in result.stdout.lower() or "target" in result.stdout.lower()


# T6-16: successful assess exits 0 and returns valid JSON
@pytest.mark.integration
def test_cli_assess_success(tmp_path: pathlib.Path) -> None:
    csv_p = _make_csv(tmp_path)
    out_p = tmp_path / "result.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modelroot",
            "assess",
            str(csv_p),
            "--target",
            "label",
            "--positive-class",
            "yes",
            "--out",
            str(out_p),
            "--seed",
            "0",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"stderr: {result.stderr}"
    assert out_p.exists()
    data = json.loads(out_p.read_text())
    assert "underperforming" in data
    assert "headline" in data


# T6-17: missing --target exits 2
def test_cli_missing_target_exits_nonzero(tmp_path: pathlib.Path) -> None:
    csv_p = _make_csv(tmp_path)
    result = subprocess.run(
        [sys.executable, "-m", "modelroot", "assess", str(csv_p)], capture_output=True, text=True
    )
    assert result.returncode != 0


# T6-18: invalid --params JSON exits 2
def test_cli_invalid_json_params_exits_2(tmp_path: pathlib.Path) -> None:
    csv_p = _make_csv(tmp_path)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modelroot",
            "assess",
            str(csv_p),
            "--target",
            "label",
            "--params",
            "{bad json}",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2


# T6-19: nonexistent CSV file exits non-zero
@pytest.mark.integration
def test_cli_nonexistent_file_exits_nonzero(tmp_path: pathlib.Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "modelroot",
            "assess",
            str(tmp_path / "nonexistent.csv"),
            "--target",
            "label",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode != 0
    # Should not expose traceback
    assert "Traceback" not in result.stdout
