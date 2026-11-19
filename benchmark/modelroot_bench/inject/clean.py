"""Clean control injector (M13).

Creates uncorrupted baseline control scenarios that record natural false-positive
rates on clean datasets.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth


class CleanControlInjector(Injector):
    """Generates clean control scenarios with no failure injections."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        # Default well-regularized baseline model
        if model_spec is None:
            model_spec = {
                "algorithm": "random_forest",
                "params": {"n_estimators": 50, "min_samples_leaf": 5, "max_depth": 6},
            }

        clean_copy = df.copy()
        gt = GroundTruth(
            injected=[],
            distractors=[],
            impact=0.0,
            no_effect=False,
        )

        return InjectedScenario(
            df=clean_copy,
            model_spec=model_spec,
            ground_truth=gt,
            clean_df=clean_copy,
        )
