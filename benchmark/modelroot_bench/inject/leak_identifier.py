"""Identifier/row-order leakage injector (M15).

Injects row-order leakage by sorting rows by target and adding an integer row_id:
- Severities: S1 (70%), S2 (90%), S3 (100% sorted by target)
- Adds integer row_id equal to row position
- Assigns 1,000 random rows to holdout using split
- Label: leakage:identifier, direction: inflation
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from modelroot.utils.seeds import SeedManager

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth, InjectedFailure

SEVERITY_SORT_FRACTIONS: dict[str, float] = {
    "S1": 0.70,
    "S2": 0.90,
    "S3": 1.00,
}


class LeakIdentifierInjector(Injector):
    """Injects row-order leakage via target sorting and row_id column."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        inj_seed = SeedManager(seed).spawn("inject:leak_identifier")
        rng = np.random.default_rng(inj_seed)
        clean_df = df.copy()

        sort_frac = SEVERITY_SORT_FRACTIONS.get(severity, 0.90)
        target_col = dataset_spec.target

        # Subsample to 3,000 rows if larger
        n_total = len(df)
        if n_total > 3000:
            sub_idx = rng.choice(n_total, size=3000, replace=False)
            df_work = df.iloc[sub_idx].copy()
        else:
            df_work = df.copy()
        n = len(df_work)

        # Sort by target
        df_sorted = df_work.sort_values(by=target_col).reset_index(drop=True)

        # Partially shuffle (1 - sort_frac) rows to match severity
        if sort_frac < 1.0:
            n_shuffle = round(n * (1.0 - sort_frac))
            if n_shuffle > 1:
                shuffle_indices = rng.choice(n, size=n_shuffle, replace=False)
                shuffled_positions = rng.permutation(shuffle_indices)
                df_sorted.iloc[shuffle_indices] = df_sorted.iloc[shuffled_positions].to_numpy()

        # Add integer row_id equal to row position
        df_sorted["row_id"] = np.arange(n, dtype=int)

        # Assign 1,000 random rows to holdout (or 1/3 if n < 3000)
        n_ho = min(1000, max(200, n // 3))
        ho_indices = set(rng.choice(n, size=n_ho, replace=False))
        splits = ["test" if i in ho_indices else "train" for i in range(n)]
        df_sorted["split"] = splits

        if model_spec is None:
            model_spec = {
                "algorithm": "random_forest",
                "params": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 1},
            }

        expected_subtypes = {
            "leakage:identifier": True,
        }
        if params and "expected_subtypes" in params:
            expected_subtypes.update(params["expected_subtypes"])

        gt = GroundTruth(
            injected=[InjectedFailure(label="leakage:identifier", severity=severity)],
            distractors=[],
            impact=0.0,
            no_effect=False,
            expected_subtypes=expected_subtypes,
        )

        return InjectedScenario(
            df=df_sorted,
            model_spec=model_spec,
            ground_truth=gt,
            clean_df=clean_df,
            direction="inflation",
        )
