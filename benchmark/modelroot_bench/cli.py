"""Benchmark command-line interface (M13).

Provides ``run`` and ``score`` subcommands for the ModelRoot benchmark suite:
    python -m modelroot_bench run --grid mini --out results/mini.jsonl
    python -m modelroot_bench score --in results/mini.jsonl --out report.json
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from modelroot_bench.grid import build_grid, load_grid_config
from modelroot_bench.io import read_benchmark_records, write_jsonl
from modelroot_bench.runner import execute_scenario
from modelroot_bench.scenario import BenchmarkRecord
from modelroot_bench.scoring import score


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelroot-bench",
        description="ModelRoot Failure Injection Benchmark CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # run subcommand
    run_p = sub.add_parser("run", help="Execute benchmark scenarios across a grid.")
    run_p.add_argument(
        "--grid",
        required=True,
        help="Grid name ('mini') or path to a grid JSON configuration file.",
    )
    run_p.add_argument(
        "--out",
        required=True,
        help="Destination path for output JSONL records (e.g. results/mini.jsonl).",
    )
    run_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of scenarios to execute.",
    )

    # score subcommand
    score_p = sub.add_parser("score", help="Score completed benchmark JSONL records.")
    score_p.add_argument(
        "--in",
        dest="in_path",
        required=True,
        help="Path to input benchmark JSONL file.",
    )
    score_p.add_argument(
        "--out",
        default=None,
        help="Optional path to write JSON score report.",
    )
    score_p.add_argument(
        "--no-bootstrap",
        action="store_true",
        default=False,
        help="Skip bootstrap confidence interval calculation.",
    )
    score_p.add_argument(
        "--seed",
        type=int,
        default=42,
        help="RNG seed for bootstrap resampling.",
    )

    return parser


def run_command(args: argparse.Namespace) -> int:
    """Execute the 'run' subcommand."""
    grid_cfg = load_grid_config(args.grid)
    specs = build_grid(grid_cfg)

    if args.limit is not None and args.limit > 0:
        specs = specs[: args.limit]

    print(f"Loaded grid with {len(specs)} scenario(s).")
    print(f"Writing outputs to: {args.out}")

    records: list[BenchmarkRecord] = []
    ctrl_cache: dict[tuple[str, str, int], dict[str, str]] = {}
    t_start = time.monotonic()

    for idx, spec in enumerate(specs, start=1):
        t0 = time.monotonic()
        print(f"[{idx}/{len(specs)}] Running scenario: {spec.id} ... ", end="", flush=True)
        rec = execute_scenario(spec, control_cache=ctrl_cache)
        records.append(rec)
        dt = time.monotonic() - t0
        st_dict = rec.result.statuses if rec.result else {}
        st_summary = ", ".join(f"{k}={v}" for k, v in st_dict.items())
        print(f"done ({dt:.2f}s) -> {st_summary}")

    write_jsonl(records, args.out)
    total_time = time.monotonic() - t_start
    print(f"Completed {len(records)} scenario(s) in {total_time:.2f}s.")
    return 0


def score_command(args: argparse.Namespace) -> int:
    """Execute the 'score' subcommand."""
    records = read_benchmark_records(args.in_path)
    print(f"Read {len(records)} benchmark record(s) from {args.in_path}.")

    rep = score(
        records,
        bootstrap=not args.no_bootstrap,
        seed=args.seed,
    )

    print()
    print(rep.summary_table())

    if args.out:
        out_p = Path(args.out)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with out_p.open("w", encoding="utf-8") as f:
            f.write(rep.model_dump_json(indent=2))
        print(f"\nSaved structured score report to: {args.out}")

    return 0


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for modelroot-bench."""
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "run":
        code = run_command(args)
    elif args.command == "score":
        code = score_command(args)
    else:
        parser.print_help()
        code = 1

    sys.exit(code)


if __name__ == "__main__":
    main()
