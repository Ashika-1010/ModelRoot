"""Synthetic dataset factories for M11 tests.

Each factory returns a CSV file path written to *tmp_path*.

The CSV always has a ``split`` column with values ``"train"`` / ``"test"``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.datasets import make_classification


def make_overfit(
    tmp_path: Path,
    n_train: int = 150,
    n_test: int = 300,
    n_noise: int = 30,
    flip_y: float = 0.10,
    seed: int = 42,
) -> Path:
    """Return a CSV designed to produce a high train-holdout gap and supported H1.

    Uses an informative signal with approximately 10% label noise and 30 Gaussian
    noise features. Both train and holdout come from the exact same underlying
    data-generating process. A high-capacity tree overfits to noise in the training
    set, while a regularized tree achieves significantly higher generalization.

    Args:
        tmp_path: Directory in which to write the CSV.
        n_train: Training-set size.
        n_test: Test-set size (>= 200 to pass the power gate).
        n_noise: Number of noise feature columns.
        flip_y: Fraction of labels to flip (label noise).
        seed: Random seed.

    Returns:
        Path to the written CSV file.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    n_total = n_train + n_test
    n_inf = 3

    X, y = make_classification(
        n_samples=n_total,
        n_features=n_inf + n_noise,
        n_informative=n_inf,
        n_redundant=0,
        n_repeated=0,
        n_classes=2,
        flip_y=flip_y,
        class_sep=1.5,
        random_state=seed,
    )

    cols: dict[str, Any] = {f"f_{i}": X[:, i].astype(str) for i in range(X.shape[1])}
    cols["label"] = ["pos" if v == 1 else "neg" for v in y]
    cols["split"] = ["train"] * n_train + ["test"] * n_test

    df = pd.DataFrame(cols)
    p = tmp_path / "overfit.csv"
    df.to_csv(p, index=False)
    return p


def make_clean(
    tmp_path: Path,
    n_train: int = 150,
    n_test: int = 300,
    sep: float = 3.0,
    seed: int = 0,
) -> Path:
    """Return a CSV with well-separated classes (no overfitting expected).

    Args:
        tmp_path: Directory in which to write the CSV.
        n_train: Training-set size.
        n_test: Test-set size.
        sep: Class separation (higher = easier).
        seed: Random seed.

    Returns:
        Path to the written CSV file.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    # Two clearly separated clusters
    half = n_total // 2
    x_pos = rng.normal(sep, 0.5, half)
    x_neg = rng.normal(-sep, 0.5, n_total - half)

    x = np.concatenate([x_pos, x_neg])
    y = np.array([1] * half + [0] * (n_total - half))

    idx = rng.permutation(n_total)
    x, y = x[idx], y[idx]

    split_col = ["train"] * n_train + ["test"] * n_test

    df = pd.DataFrame(
        {
            "feature": x.astype(str),
            "label": ["pos" if v == 1 else "neg" for v in y],
            "split": split_col,
        }
    )
    p = tmp_path / "clean.csv"
    df.to_csv(p, index=False)
    return p


def make_random_labels(
    tmp_path: Path,
    n_train: int = 150,
    n_test: int = 300,
    seed: int = 0,
) -> Path:
    """Return a CSV with random binary labels (no signal).

    Args:
        tmp_path: Directory in which to write the CSV.
        n_train: Training-set size.
        n_test: Test-set size.
        seed: Random seed.

    Returns:
        Path to the written CSV file.
    """
    tmp_path.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    n_total = n_train + n_test

    x = rng.normal(0, 1, n_total)
    y = rng.integers(0, 2, n_total)

    split_col = ["train"] * n_train + ["test"] * n_test

    df = pd.DataFrame(
        {
            "feature": x.astype(str),
            "label": ["pos" if v == 1 else "neg" for v in y],
            "split": split_col,
        }
    )
    p = tmp_path / "random_labels.csv"
    df.to_csv(p, index=False)
    return p


def make_config(
    csv_path: Path,
    algorithm: str = "decision_tree",
    algorithm_params: dict[str, Any] | None = None,
    positive_class: str = "pos",
    metric: str = "macro_f1",
    seed: int = 0,
    run_experiments: bool = True,
) -> Any:
    """Build an :class:`AnalysisConfig` for a synthetic CSV with a split column.

    Args:
        csv_path: Path to the CSV file.
        algorithm: Model algorithm name.
        algorithm_params: Optional hyper-parameter overrides.
        positive_class: Positive class label.
        metric: Primary evaluation metric.
        seed: Random seed.
        run_experiments: Whether to run experiments.

    Returns:
        :class:`AnalysisConfig` instance.
    """
    from modelroot.schemas.common import Algorithm, PrimaryMetric, SplitMode
    from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig

    return AnalysisConfig(
        data_path=csv_path,
        target="label",
        positive_class=positive_class,
        split=SplitConfig(mode=SplitMode.split_column, split_column="split"),
        model_spec=ModelSpec(
            algorithm=Algorithm(algorithm),
            params=algorithm_params or {},
        ),
        primary_metric=PrimaryMetric(metric),
        seed=seed,
        run_experiments=run_experiments,
    )
