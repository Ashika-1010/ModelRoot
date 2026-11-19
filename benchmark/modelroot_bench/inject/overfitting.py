"""H1 Overfitting failure injector (M13).

Applies the overfitting failure injection specified in §5-H1 and §6.3:
- Subsamples training partition to n in {1000, 400, 150} for severities S1, S2, S3.
- Adds 30 Gaussian noise features.
- Configures high-capacity model parameters (unpruned tree / C=1e4).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth, InjectedFailure

SEVERITY_SAMPLE_SIZES: dict[str, int] = {
    "S1": 1000,
    "S2": 400,
    "S3": 150,
}


def _get_high_capacity_spec(base_model_spec: dict[str, Any] | None) -> dict[str, Any]:
    """Derive high-capacity estimator hyperparameters for overfitting injection."""
    algo = "random_forest"
    if base_model_spec is not None and "algorithm" in base_model_spec:
        algo = str(base_model_spec["algorithm"])

    params: dict[str, Any]
    if algo == "decision_tree":
        params = {"max_depth": 30, "min_samples_leaf": 1}
    elif algo == "random_forest":
        params = {"n_estimators": 25, "min_samples_leaf": 1, "max_depth": 25}
    elif algo == "logistic_regression":
        params = {"C": 10000.0}
    elif algo == "hist_gradient_boosting":
        params = {"max_depth": 20, "min_samples_leaf": 1, "l2_regularization": 0.0}
    else:
        params = {"min_samples_leaf": 1}

    return {"algorithm": algo, "params": params}


class OverfittingInjector(Injector):
    """Injects overfitting failure via noise features, train subsampling, and high capacity."""

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
        clean_df = df.copy()

        # 1. Determine target sample sizes
        n_desired_train = SEVERITY_SAMPLE_SIZES.get(severity, 400)
        total_rows = len(df)
        target_col = dataset_spec.target

        # Reserve holdout rows (at least 200 for power gate, preferably 250-300)
        n_holdout = min(300, max(200, total_rows // 4))
        n_max_train = total_rows - n_holdout
        n_train = min(n_desired_train, n_max_train)

        # 2. Stratified train/holdout partition indices
        y_vals = np.asarray(df[target_col])
        unique_classes = np.unique(y_vals)

        train_indices: list[int] = []
        holdout_indices: list[int] = []

        for cls in unique_classes:
            cls_idx = np.where(y_vals == cls)[0]
            rng.shuffle(cls_idx)
            cls_prop = len(cls_idx) / total_rows
            n_cls_holdout = max(1, round(n_holdout * cls_prop))
            n_cls_train = max(2, round(n_train * cls_prop))

            holdout_indices.extend(cls_idx[:n_cls_holdout].tolist())
            train_indices.extend(cls_idx[n_cls_holdout : n_cls_holdout + n_cls_train].tolist())

        selected_indices = train_indices + holdout_indices
        scenario_df = df.iloc[selected_indices].copy().reset_index(drop=True)

        # Assign split column
        split_col = ["train"] * len(train_indices) + ["test"] * len(holdout_indices)
        scenario_df["split"] = split_col

        # 3. Add noise features (up to 30, keeping total columns <= 48 to respect max_cols limit)
        n_scenario_rows = len(scenario_df)
        n_noise = min(30, max(5, 48 - len(scenario_df.columns)))
        noise_matrix = rng.normal(loc=0.0, scale=1.0, size=(n_scenario_rows, n_noise))
        noise_cols = [f"noise_{i:02d}" for i in range(n_noise)]
        noise_df = pd.DataFrame(noise_matrix, columns=noise_cols, index=scenario_df.index)
        scenario_df = pd.concat([scenario_df, noise_df], axis=1)

        # 4. High capacity model spec
        high_cap_spec = _get_high_capacity_spec(model_spec)

        gt = GroundTruth(
            injected=[InjectedFailure(label="overfitting", severity=severity)],
            distractors=[],
            impact=0.0,
            no_effect=False,
        )

        return InjectedScenario(
            df=scenario_df,
            model_spec=high_cap_spec,
            ground_truth=gt,
            clean_df=clean_df,
        )
