"""Duplicate/overlap leakage injector (M15).

Injects exact duplicate rows from the training split into the holdout split:
- Severities: S1 (3%), S2 (10%), S3 (25%)
- Appends c = round(n_ho_clean * f / (1 - f)) exact training-row copies
- Label: leakage:duplicate, direction: inflation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from modelroot.utils.seeds import SeedManager

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth, InjectedFailure

SEVERITY_FRACTIONS: dict[str, float] = {
    "S1": 0.03,
    "S2": 0.10,
    "S3": 0.25,
}


class LeakDuplicateInjector(Injector):
    """Injects duplicate training rows into holdout."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        inj_seed = SeedManager(seed).spawn("inject:leak_duplicate")
        rng = np.random.default_rng(inj_seed)
        clean_df = df.copy()

        f = SEVERITY_FRACTIONS.get(severity, 0.10)
        n_total = len(df)

        # Baseline split: 2,000 train and 1,000 clean holdout rows (or 2:1 split if smaller)
        if n_total >= 3000:
            n_clean_ho = 1000
            n_tr = 2000
        else:
            n_clean_ho = max(200, n_total // 3)
            n_tr = n_total - n_clean_ho

        perm_idx = rng.permutation(n_total)
        tr_idx = perm_idx[:n_tr]
        ho_clean_idx = perm_idx[n_tr : n_tr + n_clean_ho]

        df_train = df.iloc[tr_idx].copy()
        df_ho_clean = df.iloc[ho_clean_idx].copy()

        # Calculate duplicate rows to append: c = round(n_clean_ho * f / (1 - f))
        c = round(n_clean_ho * f / (1.0 - f))
        c = max(1, min(c, len(df_train)))

        dup_sampled_idx = rng.choice(len(df_train), size=c, replace=(c > len(df_train)))
        df_ho_dups = df_train.iloc[dup_sampled_idx].copy()

        df_ho = pd.concat([df_ho_clean, df_ho_dups], ignore_index=True)

        df_train["split"] = "train"
        df_ho["split"] = "test"

        scenario_df = pd.concat([df_train, df_ho], ignore_index=True)

        if model_spec is None:
            model_spec = {
                "algorithm": "random_forest",
                "params": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 1},
            }

        expected_subtypes = {
            "leakage:duplicate": True,
        }
        if params and "expected_subtypes" in params:
            expected_subtypes.update(params["expected_subtypes"])

        gt = GroundTruth(
            injected=[InjectedFailure(label="leakage:duplicate", severity=severity)],
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
            direction="inflation",
        )
