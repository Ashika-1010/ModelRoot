"""Distribution shift failure injector (M20).

Injects distribution shift between training/holdout reference data and a current
production split:
- Severities:
  - S1: mild shift (1.0 std dev shift on top numeric feature)
  - S2: moderate shift (2.0 std dev shift on top numeric features)
  - S3: severe shift (3.5 std dev shift on numeric features + concept shift)
- Label: "shift", direction: "degradation"
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from modelroot.utils.seeds import SeedManager

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth, InjectedFailure

SEVERITY_CONFIGS: dict[str, dict[str, float]] = {
    "S1": {"shift_std": 1.0, "n_features": 1, "noise_flip": 0.05},
    "S2": {"shift_std": 2.0, "n_features": 2, "noise_flip": 0.15},
    "S3": {"shift_std": 3.5, "n_features": 3, "noise_flip": 0.30},
}


class ShiftInjector(Injector):
    """Injects dataset covariate and concept drift into current split."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        inj_seed = SeedManager(seed).spawn("inject:shift")
        rng = np.random.default_rng(inj_seed)

        cfg = SEVERITY_CONFIGS.get(severity, SEVERITY_CONFIGS["S2"])
        target_col = dataset_spec.target
        clean_df = df.copy()

        if model_spec is None:
            model_spec = {
                "algorithm": "random_forest",
                "params": {"n_estimators": 50, "max_depth": 8},
            }

        n_total = len(df)
        # Split reference (train + holdout) and current
        # If dataset >= 3000 rows, use 1000 for current
        if n_total >= 3000:
            n_curr = 1000
            n_ref = n_total - n_curr
        else:
            n_curr = max(100, n_total // 3)
            n_ref = n_total - n_curr

        perm = rng.permutation(n_total)
        ref_idx = perm[:n_ref]
        curr_idx = perm[n_ref:]

        df_ref = df.iloc[ref_idx].copy().reset_index(drop=True)
        df_curr = df.iloc[curr_idx].copy().reset_index(drop=True)

        # Identify continuous numeric features excluding target and split
        num_cols = [
            c
            for c in df.columns
            if c != target_col
            and c != "split"
            and pd.api.types.is_numeric_dtype(df[c])
            and not pd.api.types.is_bool_dtype(df[c])
        ]

        n_shift_feats = int(min(cfg["n_features"], len(num_cols)))
        shifted_cols: list[str] = []

        if num_cols:
            selected_cols = list(rng.choice(num_cols, size=n_shift_feats, replace=False))
            shifted_cols.extend(selected_cols)

            for col in selected_cols:
                col_std = float(df_ref[col].std())
                if np.isnan(col_std) or col_std == 0:
                    col_std = 1.0
                shift_val = cfg["shift_std"] * col_std
                # Shift current data feature distribution
                df_curr[col] = df_curr[col] + shift_val

        # Label noise in shifted regime to induce error increase
        noise_flip = cfg["noise_flip"]
        if noise_flip > 0 and target_col in df_curr.columns:
            y_curr = df_curr[target_col].to_numpy().copy()
            flip_mask = rng.uniform(0, 1, size=len(y_curr)) < noise_flip
            y_curr[flip_mask] = 1 - y_curr[flip_mask]
            df_curr[target_col] = y_curr

        gt = GroundTruth(
            injected=[InjectedFailure(label="shift", severity=severity)],
            impact=0.10,
            no_effect=False,
        )

        return InjectedScenario(
            df=df_ref,
            model_spec=model_spec,
            ground_truth=gt,
            clean_df=clean_df,
            current_df=df_curr,
            direction="degradation",
            metadata={},
        )
