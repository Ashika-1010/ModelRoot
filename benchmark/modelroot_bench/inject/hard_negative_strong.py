"""Hard-negative strong feature injector (M16).

Injects a legitimately strong predictor into both train and holdout splits.
The feature IS a strong predictor but is NOT leakage — it's available at inference time.

This verifies that H3c does not false-positive on a legitimate strong feature
that is not flagged as unavailable at inference.

Label: empty (no failure injected), direction: degradation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from modelroot.utils.seeds import SeedManager

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth


class HardNegativeStrongInjector(Injector):
    """Injects a legitimate strong predictor — not leakage.

    Adds 'strong_pred = y + small_noise' to both train and holdout splits.
    The feature is not flagged as unavailable, so H3c should not confirm it.
    """

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
        inj_seed = SeedManager(seed).spawn("inject:hard_negative_strong")
        rng = np.random.default_rng(inj_seed)

        target_col = dataset_spec.target
        clean_df = df.copy()

        if model_spec is None:
            model_spec = {
                "algorithm": "decision_tree",
                "params": {"max_depth": 6},
            }

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

        # Add a strong but legitimate predictor (available at inference time)
        y_tr = df_train[target_col].to_numpy(dtype=float)
        y_ho_arr = df_ho[target_col].to_numpy(dtype=float)
        df_train["strong_pred"] = y_tr + rng.standard_normal(len(y_tr)) * 0.3
        df_ho["strong_pred"] = y_ho_arr + rng.standard_normal(len(y_ho_arr)) * 0.3

        df_train["split"] = "train"
        df_ho["split"] = "test"
        scenario_df = pd.concat([df_train, df_ho], ignore_index=True)

        # No failures injected — clean hard negative
        gt = GroundTruth(
            injected=[],
            distractors=[],
            impact=0.0,
            no_effect=True,
            expected_subtypes={},
        )

        return InjectedScenario(
            df=scenario_df,
            model_spec=model_spec,
            ground_truth=gt,
            clean_df=clean_df,
            current_df=None,
            direction="degradation",
            metadata={},
        )
