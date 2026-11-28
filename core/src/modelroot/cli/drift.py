"""``modelroot drift`` CLI subcommand (M19).

Usage::

    modelroot drift --reference REF.csv --current CUR.csv [--target COL] [--out OUT.json]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

from modelroot.stats.drift import compute_drift_report


def run_drift_cli(args: argparse.Namespace) -> None:
    """Execute the ``drift`` sub-command."""
    ref_path = Path(args.reference)
    cur_path = Path(args.current)

    if not ref_path.exists():
        print(f"Error: reference file not found: {ref_path}", file=sys.stderr)
        sys.exit(2)

    if not cur_path.exists():
        print(f"Error: current file not found: {cur_path}", file=sys.stderr)
        sys.exit(2)

    try:
        df_ref = pd.read_csv(ref_path)
    except Exception as exc:
        print(f"Error reading reference CSV: {exc}", file=sys.stderr)
        sys.exit(2)

    try:
        df_cur = pd.read_csv(cur_path)
    except Exception as exc:
        print(f"Error reading current CSV: {exc}", file=sys.stderr)
        sys.exit(2)

    n_perms = getattr(args, "n_permutations", 50)
    seed = getattr(args, "seed", 0)
    target = getattr(args, "target", None)

    report = compute_drift_report(
        df_ref=df_ref,
        df_cur=df_cur,
        target_column=target,
        seed=seed,
        n_permutations=n_perms,
    )

    # Print human-readable summary
    print("=" * 60)
    print("ModelRoot Dataset Drift Report (M19)")
    print("=" * 60)
    print(f"Reference observations: {report.n_ref}")
    print(f"Current observations:   {report.n_cur}")
    print("-" * 60)
    print("Multidimensional Domain Classifier:")
    print(f"  Pooled OOF ROC-AUC:    {report.domain_classifier.auc:.4f}")
    print(f"  Permutation p-value:   {report.domain_classifier.p_value:.4f}")
    drift_detected = (
        report.domain_classifier.auc >= 0.60 and report.domain_classifier.p_value < 0.01
    )
    verdict_str = "SIGNIFICANT DRIFT DETECTED" if drift_detected else "NO SIGNIFICANT DRIFT"
    print(f"  Verdict:               {verdict_str}")
    print("-" * 60)
    print("Top Drifted Features (by PSI):")
    print(f"{'Feature':<25} {'Type':<12} {'PSI':<10} {'Stat (p-val)':<18}")
    print("-" * 60)

    for feat in report.top_drifted_features[:10]:
        u = report.univariate[feat]
        stat_desc = f"{u.statistic_name}={u.statistic_value:.3f} (p={u.p_value:.3g})"
        print(f"{feat:<25} {u.feature_type:<12} {u.psi:<10.4f} {stat_desc:<18}")

    print("=" * 60)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report.to_dict(), f, indent=2)
        print(f"Saved drift report JSON to: {out_path}")

    sys.exit(0)
