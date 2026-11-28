"""H4 Subgroup underperformance failure injector (M18).

Applies subgroup failure injection specified in §5-H4 and §6.3:
- Selects or creates a feature-defined subgroup segment (5%, 15%, 30% by severity).
- Corrupts labels selectively inside the slice to induce localized error degradation.
- Leaves complement performance uncorrupted.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth, InjectedFailure

SEVERITY_CONFIG: dict[str, dict[str, float]] = {
    "S1": {"slice_fraction": 0.30, "error_prob": 0.45},
    "S2": {"slice_fraction": 0.20, "error_prob": 0.65},
    "S3": {"slice_fraction": 0.15, "error_prob": 0.85},
}


class SubgroupInjector(Injector):
    """Injects localized underperformance into a specific subgroup slice."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        rng = np.random.default_rng(seed)
        df_out = df.copy()
        clean_df = df.copy()

        cfg = SEVERITY_CONFIG.get(severity, SEVERITY_CONFIG["S2"])
        target_col = dataset_spec.target

        # Select or inject a subgroup marker feature
        # If dataset has categorical columns, use one, otherwise inject 'user_segment'
        cat_cols = [
            c
            for c in df_out.columns
            if c != target_col
            and c != "split"
            and df_out[c].dtype in (object, "category")
            and df_out[c].nunique() >= 2
        ]

        if cat_cols:
            seg_col = cat_cols[0]
            top_val = df_out[seg_col].value_counts().index[0]
            mask = (df_out[seg_col] == top_val).to_numpy()
        else:
            # Inject a clean categorical segment feature
            seg_col = "user_segment"
            n_rows = len(df_out)
            n_slice = max(35, int(n_rows * cfg["slice_fraction"]))
            seg_vals = np.array(["control"] * n_rows, dtype=object)
            slice_indices = rng.choice(n_rows, size=n_slice, replace=False)
            seg_vals[slice_indices] = "underperforming"
            df_out[seg_col] = seg_vals
            clean_df[seg_col] = seg_vals
            mask = seg_vals == "underperforming"

        # Corrupt labels inside the subgroup slice
        y = df_out[target_col].to_numpy().copy()
        slice_idx = np.where(mask)[0]

        flip_mask = rng.uniform(0.0, 1.0, size=len(slice_idx)) < cfg["error_prob"]
        to_flip = slice_idx[flip_mask]
        # Binary target flip (0 -> 1, 1 -> 0)
        y[to_flip] = 1 - y[to_flip]
        df_out[target_col] = y

        final_spec = (
            model_spec
            if model_spec is not None
            else {"algorithm": "random_forest", "params": {"n_estimators": 50, "max_depth": 10}}
        )

        gt = GroundTruth(
            injected=[InjectedFailure(label="subgroup", severity=severity)],
            impact=0.08,
            no_effect=False,
        )

        return InjectedScenario(
            df=df_out,
            model_spec=final_spec,
            ground_truth=gt,
            clean_df=clean_df,
            metadata={"sensitive_columns": [seg_col]},
        )
