"""``modelroot assess`` CLI entry point (M6).

Usage::

    modelroot assess DATA --target COLUMN [options]

Exit codes:
  0 — success
  1 — internal error
  2 — contract violation / invalid input
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="modelroot",
        description="ModelRoot — automated ML failure diagnosis engine.",
    )
    sub = parser.add_subparsers(dest="command")

    # ------------------------------------------------------------------
    # Shared argument builder
    # ------------------------------------------------------------------
    def _add_common_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("data", metavar="DATA", help="Path to the primary CSV dataset.")
        p.add_argument("--target", required=True, help="Target column name.")
        p.add_argument("--positive-class", default="1", help="Positive class (default: '1').")
        p.add_argument(
            "--algorithm",
            default="random_forest",
            choices=[
                "logistic_regression",
                "decision_tree",
                "random_forest",
                "hist_gradient_boosting",
            ],
            help="Estimator algorithm (default: random_forest).",
        )
        p.add_argument("--params", default="{}", help="JSON-encoded hyper-parameters.")
        p.add_argument("--test", default=None, help="Separate test CSV file.")
        p.add_argument("--current", default=None, help="Current-data CSV file.")
        p.add_argument(
            "--metric",
            default="macro_f1",
            choices=["accuracy", "balanced_accuracy", "macro_f1", "f1", "roc_auc", "pr_auc"],
            help="Primary metric (default: macro_f1).",
        )
        p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
        p.add_argument("--out", default=None, help="Path to write JSON output.")

    assess_p = sub.add_parser("assess", help="Assess whether a model is underperforming.")
    _add_common_args(assess_p)

    diagnose_p = sub.add_parser("diagnose", help="Run full failure diagnosis with H1.")
    _add_common_args(diagnose_p)
    diagnose_p.add_argument(
        "--no-experiments",
        action="store_true",
        default=False,
        help="Skip experiment fitting (detector-only mode).",
    )
    diagnose_p.add_argument(
        "--rules-profile",
        default="full",
        choices=["full", "naive"],
        help="Decision-rule profile (default: full).",
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    """CLI entry point.

    Args:
        argv: Argument list; uses ``sys.argv[1:]`` if None.
    """
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    if args.command == "assess":
        _run_assess(args)
    elif args.command == "diagnose":
        _run_diagnose(args)


def _run_assess(args: argparse.Namespace) -> None:
    """Execute the ``assess`` sub-command.

    Args:
        args: Parsed argument namespace.
    """
    try:
        params: dict[str, int | float | str | bool | None] = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Error: --params is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)

    # Build config
    try:
        from modelroot.assessment.verdict import assess_only
        from modelroot.config.thresholds import Thresholds
        from modelroot.schemas.common import (
            Algorithm,
            PrimaryMetric,
            SplitMode,
        )
        from modelroot.schemas.config import (
            AnalysisConfig,
            ModelSpec,
            SplitConfig,
        )

        split_mode = SplitMode.test_file if args.test else SplitMode.single_file
        split_kwargs: dict[str, object] = {"mode": split_mode}
        if args.test:
            split_kwargs["test_path"] = args.test

        algorithm = Algorithm(args.algorithm)
        config = AnalysisConfig(
            data_path=Path(args.data),
            target=args.target,
            positive_class=args.positive_class,
            split=SplitConfig(**split_kwargs),  # type: ignore[arg-type]
            current_path=Path(args.current) if args.current else None,
            model_spec=ModelSpec(algorithm=algorithm, params=params),
            primary_metric=PrimaryMetric(args.metric),
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        result = assess_only(config, thresholds=Thresholds())
    except Exception as exc:
        # Distinguish contract violations from internal errors
        from modelroot.data.errors import ContractViolation

        if isinstance(exc, ContractViolation):
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"Internal error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Human-readable summary
    pm = config.primary_metric.value
    print("\n=== ModelRoot Assessment ===")
    print(f"Dataset   : {args.data}")
    print(f"Algorithm : {args.algorithm}")
    print(f"Metric    : {pm}")
    print(f"Headline  : {result.headline}")
    print(f"Underperforming: {result.underperforming}")
    if result.reasons:
        print("Reasons:")
        for r in result.reasons:
            print(f"  - {r}")

    # JSON output
    result_json = result.model_dump(mode="json")

    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(result_json, indent=2, default=str), encoding="utf-8")
        print(f"\nResults written to {out_path}")
    else:
        print("\n--- JSON Output ---")
        print(json.dumps(result_json, indent=2, default=str))

    sys.exit(0)


if __name__ == "__main__":
    main()


def _build_config_from_args(args: argparse.Namespace) -> object:
    """Build ``AnalysisConfig`` from parsed args (shared by assess and diagnose)."""
    try:
        params: dict[str, int | float | str | bool | None] = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Error: --params is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)
    try:
        from modelroot.schemas.common import Algorithm, PrimaryMetric, SplitMode
        from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig

        split_mode = SplitMode.test_file if args.test else SplitMode.single_file
        split_kwargs: dict[str, object] = {"mode": split_mode}
        if args.test:
            split_kwargs["test_path"] = args.test
        algorithm = Algorithm(args.algorithm)
        return AnalysisConfig(
            data_path=Path(args.data),
            target=args.target,
            positive_class=args.positive_class,
            split=SplitConfig(**split_kwargs),  # type: ignore[arg-type]
            current_path=Path(args.current) if args.current else None,
            model_spec=ModelSpec(algorithm=algorithm, params=params),
            primary_metric=PrimaryMetric(args.metric),
            seed=args.seed,
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)


def _run_diagnose(args: argparse.Namespace) -> None:
    """Execute the ``diagnose`` sub-command."""
    try:
        params: dict[str, int | float | str | bool | None] = json.loads(args.params)
    except json.JSONDecodeError as exc:
        print(f"Error: --params is not valid JSON: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        from modelroot.schemas.common import Algorithm, PrimaryMetric, RulesProfile, SplitMode
        from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig

        split_mode = SplitMode.test_file if args.test else SplitMode.single_file
        split_kwargs: dict[str, object] = {"mode": split_mode}
        if args.test:
            split_kwargs["test_path"] = args.test
        algorithm = Algorithm(args.algorithm)
        config = AnalysisConfig(
            data_path=Path(args.data),
            target=args.target,
            positive_class=args.positive_class,
            split=SplitConfig(**split_kwargs),  # type: ignore[arg-type]
            current_path=Path(args.current) if args.current else None,
            model_spec=ModelSpec(algorithm=algorithm, params=params),
            primary_metric=PrimaryMetric(args.metric),
            seed=args.seed,
            run_experiments=not args.no_experiments,
            rules_profile=RulesProfile(args.rules_profile),
        )
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        from modelroot.config.thresholds import Thresholds
        from modelroot.engine.pipeline import diagnose

        result = diagnose(config, thresholds=Thresholds())
    except Exception as exc:
        from modelroot.data.errors import ContractViolation

        if isinstance(exc, ContractViolation):
            print(f"Error: {exc}", file=sys.stderr)
            sys.exit(2)
        print(f"Internal error: {exc}", file=sys.stderr)
        sys.exit(1)

    # Human-readable summary
    pm = config.primary_metric.value
    print("\n=== ModelRoot Diagnosis ===")
    print(f"Dataset   : {args.data}")
    print(f"Algorithm : {args.algorithm}")
    print(f"Metric    : {pm}")
    print(f"Headline  : {result.assessment.headline}")

    if result.hypotheses:
        print("\n# | hypothesis      | status           | delta  | severity")
        print("-" * 65)
        for i, h in enumerate(result.hypotheses, 1):
            delta_str = f"{h.effect.delta:+.3f}" if h.effect else "  n/a "
            sev = h.severity.value if h.severity else "—"
            print(f"{i:2d}| {h.type.value:<16} | {h.status.value:<16} | {delta_str} | {sev}")

    if result.warnings:
        print("\nWarnings:")
        for w in result.warnings:
            print(f"  - {w}")

    print(f"\nAnalysis status: {result.status}")

    result_json = result.model_dump(mode="json")
    if args.out:
        out_path = Path(args.out)
        try:
            out_path.write_text(json.dumps(result_json, indent=2, default=str), encoding="utf-8")
            print(f"Results written to {out_path}")
        except Exception as exc:
            print(f"Error writing output: {exc}", file=sys.stderr)
            sys.exit(1)

    sys.exit(0)
