"""H2 Class Imbalance failure injector (M14).

Injects class imbalance failures by subsampling the minority class:
- Severity S1: 1:5 ratio (~16.7% minority)
- Severity S2: 1:20 ratio (~4.8% minority)
- Severity S3: 1:100 ratio (~1.0% minority)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.scenario import GroundTruth, InjectedFailure

SEVERITY_RATIOS: dict[str, int] = {
    "S1": 5,
    "S2": 20,
    "S3": 100,
}


class ImbalanceInjector(Injector):
    """Injects class imbalance by subsampling the minority class."""

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

        target_col = dataset_spec.target
        ratio = SEVERITY_RATIOS.get(severity, 5)

        # Identify positive/minority and majority classes
        y_vals = np.asarray(df[target_col])
        classes, class_counts = np.unique(y_vals, return_counts=True)
        if len(classes) != 2:
            raise ValueError(
                f"ImbalanceInjector requires binary target, found {len(classes)} classes"
            )

        # Pick minority as positive class or smaller count
        pos_val = int(dataset_spec.positive) if str(dataset_spec.positive).isdigit() else classes[1]
        min_class = pos_val if pos_val in classes else classes[np.argmin(class_counts)]
        maj_class = classes[0] if classes[1] == min_class else classes[1]

        maj_indices = np.where(y_vals == maj_class)[0]
        min_indices = np.where(y_vals == min_class)[0]
        rng.shuffle(maj_indices)
        rng.shuffle(min_indices)

        # Reserve holdout rows: at least 200 rows for power gate
        total_rows = len(df)
        n_holdout = min(300, max(200, total_rows // 4))

        n_ho_min = max(2, round(n_holdout / (ratio + 1)))
        n_ho_maj = n_holdout - n_ho_min

        # Ensure we do not request more samples than available
        n_ho_maj = min(n_ho_maj, len(maj_indices) - 10)
        n_ho_min = min(n_ho_min, len(min_indices) - 5)

        ho_maj_idx = maj_indices[:n_ho_maj]
        ho_min_idx = min_indices[:n_ho_min]

        # Remaining pool for training
        rem_maj_idx = maj_indices[n_ho_maj:]
        rem_min_idx = min_indices[n_ho_min:]

        # Train counts matching severity ratio
        n_tr_maj = len(rem_maj_idx)
        n_tr_min = max(2, min(len(rem_min_idx), n_tr_maj // ratio))

        tr_maj_idx = rem_maj_idx[:n_tr_maj]
        tr_min_idx = rem_min_idx[:n_tr_min]

        train_indices = np.concatenate([tr_maj_idx, tr_min_idx])
        holdout_indices = np.concatenate([ho_maj_idx, ho_min_idx])

        # Shuffle within splits
        rng.shuffle(train_indices)
        rng.shuffle(holdout_indices)

        scenario_df = pd.concat(
            [df.iloc[train_indices], df.iloc[holdout_indices]], ignore_index=True
        )
        split_col = ["train"] * len(train_indices) + ["test"] * len(holdout_indices)
        scenario_df["split"] = split_col

        if model_spec is None:
            model_spec = {
                "algorithm": "random_forest",
                "params": {"n_estimators": 50, "max_depth": 8, "min_samples_leaf": 1},
            }

        if params and "expected_subtypes" in params:
            expected_subtypes = dict(params["expected_subtypes"])
        elif severity == "S1":
            expected_subtypes = {
                "imbalance:distortion": True,
                "imbalance:harm": True,
            }
        elif severity in ("S2", "S3"):
            expected_subtypes = {"imbalance:distortion": True}
        else:
            expected_subtypes = {"imbalance:distortion": True}

        gt = GroundTruth(
            injected=[InjectedFailure(label="imbalance", severity=severity)],
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
        )
