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

    assess_p = sub.add_parser("assess", help="Assess whether a model is underperforming.")
    assess_p.add_argument("data", metavar="DATA", help="Path to the primary CSV dataset.")
    assess_p.add_argument("--target", required=True, help="Target column name.")
    assess_p.add_argument(
        "--positive-class", default="1", help="Positive class label (default: '1')."
    )
    assess_p.add_argument(
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
    assess_p.add_argument("--params", default="{}", help="JSON-encoded hyper-parameters.")
    assess_p.add_argument("--test", default=None, help="Path to a separate test CSV file.")
    assess_p.add_argument("--current", default=None, help="Path to a current-data CSV file.")
    assess_p.add_argument(
        "--metric",
        default="macro_f1",
        choices=["accuracy", "balanced_accuracy", "macro_f1", "f1", "roc_auc", "pr_auc"],
        help="Primary evaluation metric (default: macro_f1).",
    )
    assess_p.add_argument("--seed", type=int, default=0, help="Random seed (default: 0).")
    assess_p.add_argument("--out", default=None, help="Path to write JSON output.")
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
