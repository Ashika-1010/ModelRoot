"""Hard negative high cardinality injector (M15).

Injects a unique random string token unrelated to the target:
- High unique ratio triggers S1 trigger
- Token has no predictive power, importance near zero, status rejected
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from modelroot.utils.seeds import SeedManager

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth


class HardNegativeHighcardInjector(Injector):
    """Injects a unique random string token feature unrelated to target."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        inj_seed = SeedManager(seed).spawn("inject:hard_negative_highcard")
        rng = np.random.default_rng(inj_seed)
        clean_df = df.copy()
        scenario_df = df.copy()

        # Add unique random string token (unique ratio = 1.0)
        n = len(scenario_df)
        rand_ints = rng.integers(0, 1000000000, size=n)
        tokens = [f"tok_{i}_{rand_ints[i]}" for i in range(n)]
        scenario_df["token"] = tokens

        # Stratified or random 80/20 train/test split if split column absent
        if "split" not in scenario_df.columns:
            n_ho = min(1000, max(200, n // 4))
            ho_idx = set(rng.choice(n, size=n_ho, replace=False))
            scenario_df["split"] = ["test" if i in ho_idx else "train" for i in range(n)]

        if model_spec is None:
            model_spec = {
                "algorithm": "random_forest",
                "params": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 1},
            }

        expected_subtypes = {
            "leakage:duplicate": False,
            "leakage:identifier": False,
        }
        if params and "expected_subtypes" in params:
            expected_subtypes.update(params["expected_subtypes"])

        gt = GroundTruth(
            injected=[],
            distractors=["high_cardinality_token"],
            impact=0.0,
            no_effect=False,
            expected_subtypes=expected_subtypes,
        )

        return InjectedScenario(
            df=scenario_df,
            model_spec=model_spec,
            ground_truth=gt,
            clean_df=clean_df,
            direction="degradation",
        )
