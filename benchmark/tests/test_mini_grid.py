"""End-to-end mini grid benchmark integration tests (M13).

Executes the mini grid through the actual ModelRoot pipeline, validates contracts,
measures total runtime (< 3 minutes), and scores the first measured H1 benchmark number.
"""

from __future__ import annotations

import pathlib
import subprocess
import sys
import time

import pytest

from modelroot_bench.grid import build_grid, load_grid_config
from modelroot_bench.io import read_benchmark_records, write_jsonl
from modelroot_bench.runner import execute_scenario
from modelroot_bench.scenario import BenchmarkRecord
from modelroot_bench.scoring import score


@pytest.mark.bench
def test_mini_grid_end_to_end_scoring(tmp_path: pathlib.Path) -> None:
    """Execute the mini benchmark grid and produce the first measured H1 report."""
    cfg = load_grid_config("mini")
    specs = build_grid(cfg)
    assert len(specs) == 12

    ctrl_cache: dict[tuple[str, str, int], dict[str, str]] = {}
    records: list[BenchmarkRecord] = []

    t0 = time.monotonic()
    for spec in specs:
        rec = execute_scenario(spec, control_cache=ctrl_cache)
        assert rec.result is not None
        assert rec.result.scenario_id == spec.id
        assert isinstance(rec.result.statuses, dict)
        assert rec.result.runtime_s > 0.0
        records.append(rec)

    total_time = time.monotonic() - t0
    # Must complete in under 3 minutes (180 seconds)
    assert total_time < 180.0, f"Mini grid took {total_time:.2f}s, exceeding 180s threshold."

    # Validate JSONL roundtrip
    out_file = tmp_path / "mini_results.jsonl"
    write_jsonl(records, out_file)
    loaded_recs = read_benchmark_records(out_file)
    assert len(loaded_recs) == len(records)

    # Score and print report
    report = score(loaded_recs, bootstrap=True, n_boot=200, seed=42)
    assert report.n_scenarios == 12

    table = report.summary_table()
    print("\n" + table)

    # Acceptance: H1 numbers printed and present in report
    assert "overfitting" in report.lenient_metrics
    of_lenient = report.lenient_metrics["overfitting"]
    assert of_lenient.tp >= 0
    assert of_lenient.precision is not None or of_lenient.tp + of_lenient.fp == 0
    assert of_lenient.recall is not None or of_lenient.tp + of_lenient.fn == 0


@pytest.mark.integration
def test_cli_run_and_score(tmp_path: pathlib.Path) -> None:
    """Validate CLI run and score commands on a 2-scenario slice."""
    results_path = tmp_path / "cli_results.jsonl"
    report_path = tmp_path / "cli_report.json"

    # 1. run command
    run_cmd = [
        sys.executable,
        "-m",
        "modelroot_bench",
        "run",
        "--grid",
        "mini",
        "--limit",
        "2",
        "--out",
        str(results_path),
    ]
    res_run = subprocess.run(run_cmd, capture_output=True, text=True, cwd=tmp_path)
    assert res_run.returncode == 0, f"CLI run failed:\n{res_run.stderr}\n{res_run.stdout}"
    assert results_path.is_file()

    # 2. score command
    score_cmd = [
        sys.executable,
        "-m",
        "modelroot_bench",
        "score",
        "--in",
        str(results_path),
        "--out",
        str(report_path),
        "--no-bootstrap",
    ]
    res_score = subprocess.run(score_cmd, capture_output=True, text=True, cwd=tmp_path)
    assert res_score.returncode == 0, f"CLI score failed:\n{res_score.stderr}\n{res_score.stdout}"
    assert report_path.is_file()
    assert "ModelRoot Benchmark Score Report" in res_score.stdout
