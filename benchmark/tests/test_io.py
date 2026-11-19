"""Tests for JSONL benchmark I/O (M12)."""

from __future__ import annotations

import pathlib

import pytest

from modelroot_bench.io import read_benchmark_records, read_jsonl, write_jsonl
from modelroot_bench.scenario import (
    BenchmarkRecord,
    GroundTruth,
    ScenarioResult,
    ScenarioSpec,
)


def test_jsonl_roundtrip_benchmark_record(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "test.jsonl"
    spec = ScenarioSpec(
        id="s1",
        dataset="synthetic",
        model_spec={"algorithm": "logistic_regression", "params": {}},
        scenario_class="clean_control",
        injections=[],
        seed=10,
        role="dev",
    )
    gt = GroundTruth(injected=[], impact=0.0, no_effect=False)
    res = ScenarioResult(
        scenario_id="s1",
        statuses={"overfitting": "not_triggered"},
        runtime_s=0.5,
    )
    record = BenchmarkRecord(spec=spec, ground_truth=gt, result=res)

    write_jsonl([record], p)
    loaded = read_benchmark_records(p)
    assert len(loaded) == 1
    assert loaded[0].spec.id == "s1"
    assert loaded[0].ground_truth.injected == []
    assert loaded[0].result is not None
    assert loaded[0].result.statuses["overfitting"] == "not_triggered"


def test_read_jsonl_missing_file_raises(tmp_path: pathlib.Path) -> None:
    with pytest.raises(FileNotFoundError):
        read_jsonl(tmp_path / "nonexistent.jsonl")


def test_read_jsonl_invalid_json_raises(tmp_path: pathlib.Path) -> None:
    p = tmp_path / "bad.jsonl"
    p.write_text("{bad json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Invalid JSON"):
        read_jsonl(p)
