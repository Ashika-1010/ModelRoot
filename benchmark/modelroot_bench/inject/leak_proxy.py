"""Target-proxy leakage injector (M16).

Injects a noisy copy of the target label as a feature:
- S1: proxy = y + N(0, 1.0^2) — weak proxy
- S2: proxy = y + N(0, 0.3^2) — strong proxy
- S3: proxy = y exactly (sigma=0) — perfect proxy
- S3t: proxy corrupted in current split (sigma=2.0) — transfer variant
- S3f: proxy flagged via metadata.unavailable_at_inference

Label: leakage:target_proxy, direction: inflation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from modelroot.utils.seeds import SeedManager

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth, InjectedFailure

SEVERITY_SIGMAS: dict[str, float] = {
    "S1": 1.0,
    "S2": 0.3,
    "S3": 0.0,
    "S3t": 0.0,  # transfer variant
    "S3f": 0.0,  # flagged variant
}


class LeakProxyInjector(Injector):
    """Injects a noisy proxy of the target label as a feature."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        params = params or {}
        transfer_variant = severity == "S3t" or bool(params.get("transfer", False))
        flagged_variant = severity == "S3f" or bool(params.get("flagged", False))

        inj_seed = SeedManager(seed).spawn("inject:leak_proxy")
        rng = np.random.default_rng(inj_seed)

        sigma = SEVERITY_SIGMAS.get(severity, 0.0)
        target_col = dataset_spec.target
        clean_df = df.copy()

        if model_spec is None:
            model_spec = {
                "algorithm": "decision_tree",
                "params": {"max_depth": 6},
            }

        # Determine split structure
        # We build a train+holdout scenario df; a 'current' split is added for transfer variants
        n_total = len(df)
        if n_total >= 3000:
            n_tr = 2000
            n_ho = 1000
        else:
            n_ho = max(200, n_total // 3)
            n_tr = n_total - n_ho

        perm = rng.permutation(n_total)
        tr_idx = perm[:n_tr]
        ho_idx = perm[n_tr : n_tr + n_ho]

        df_train = df.iloc[tr_idx].copy().reset_index(drop=True)
        df_ho = df.iloc[ho_idx].copy().reset_index(drop=True)

        # Add proxy to train and holdout
        y_tr = df_train[target_col].to_numpy(dtype=float)
        y_ho_arr = df_ho[target_col].to_numpy(dtype=float)

        df_train["proxy"] = y_tr + rng.standard_normal(len(y_tr)) * sigma
        df_ho["proxy"] = y_ho_arr + rng.standard_normal(len(y_ho_arr)) * sigma

        df_train["split"] = "train"
        df_ho["split"] = "test"
        scenario_df = pd.concat([df_train, df_ho], ignore_index=True)

        # Build current_df for transfer variant
        current_df: pd.DataFrame | None = None
        if transfer_variant:
            # Use held-back rows or reuse holdout rows with highly corrupted proxy
            if n_total >= 3000 and n_total > n_tr + n_ho:
                curr_idx = perm[n_tr + n_ho :]
            else:
                curr_idx = perm[n_tr : n_tr + min(n_ho, n_total - n_tr)]

            df_curr = df.iloc[curr_idx].copy().reset_index(drop=True)
            y_curr = df_curr[target_col].to_numpy(dtype=float)
            # Transfer variant: proxy has high noise in current set (poor signal)
            df_curr["proxy"] = y_curr + rng.standard_normal(len(y_curr)) * 2.0
            df_curr["split"] = "current"
            current_df = df_curr

        expected_subtypes = {"target_proxy": True}
        metadata: dict[str, Any] = {}
        if flagged_variant:
            metadata["unavailable_at_inference"] = ["proxy"]

        gt = GroundTruth(
            injected=[InjectedFailure(label="leakage:target_proxy", severity=severity)],
            distractors=[],
            impact=0.0,
            no_effect=False,
            expected_subtypes=expected_subtypes,
        )

        return InjectedScenario(
            df=scenario_df,
            model_spec=model_spec,
            ground_truth=gt,
            clean_df=clean_df,
            current_df=current_df,
            direction="inflation",
            metadata=metadata,
        )
