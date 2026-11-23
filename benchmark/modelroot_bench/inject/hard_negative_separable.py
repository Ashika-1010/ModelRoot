"""H2 Separable hard-negative injector (M14).

Generates an imbalanced dataset (1:10 ratio) with high class separation (class_sep=3.0)
so that classification performance is uncorrupted and harm is rejected.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.datasets import make_classification

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth


class HardNegativeSeparableInjector(Injector):
    """Generates separable hard negative scenarios with 1:10 imbalance and class_sep=3.0."""

    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        target_col = dataset_spec.target
        feature_cols = [c for c in df.columns if c not in (target_col, "split")]
        n_features = len(feature_cols)

        # Generate imbalanced separable data (1:10 ratio, class_sep=3.0)
        X_scen, y_scen = make_classification(
            n_samples=len(df),
            n_features=n_features,
            n_informative=min(6, n_features),
            n_redundant=0,
            n_clusters_per_class=1,
            weights=[10 / 11, 1 / 11],
            class_sep=3.0,
            flip_y=0.0,
            random_state=seed,
        )

        scenario_df = pd.DataFrame(X_scen, columns=feature_cols)
        scenario_df[target_col] = y_scen.astype(int)

        # 80/20 train/test split
        train_size = round(len(df) * 0.8)
        split_col = ["train"] * train_size + ["test"] * (len(df) - train_size)
        scenario_df["split"] = split_col

        # Balanced reference dataset for clean baseline
        X_clean, y_clean = make_classification(
            n_samples=len(df),
            n_features=n_features,
            n_informative=min(6, n_features),
            n_redundant=0,
            n_clusters_per_class=1,
            weights=[0.5, 0.5],
            class_sep=3.0,
            flip_y=0.0,
            random_state=seed,
        )
        clean_df = pd.DataFrame(X_clean, columns=feature_cols)
        clean_df[target_col] = y_clean.astype(int)

        if model_spec is None:
            model_spec = {
                "algorithm": "random_forest",
                "params": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 1},
            }

        gt = GroundTruth(
            injected=[],
            distractors=["separable_imbalance"],
            impact=0.0,
            no_effect=True,
            expected_subtypes={
                "imbalance:distortion": False,
                "imbalance:harm": False,
            },
        )

        return InjectedScenario(
            df=scenario_df,
            model_spec=model_spec,
            ground_truth=gt,
            clean_df=clean_df,
        )
