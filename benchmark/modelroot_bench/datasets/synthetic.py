"""Built-in offline datasets for benchmarking (M13).

Generates deterministic synthetic datasets via scikit-learn and loads the
bundled breast cancer dataset. Does not require network access.
"""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import load_breast_cancer, make_classification

from modelroot_bench.datasets.registry import DatasetSpec, register_dataset


def make_synthetic_dataset(
    n_samples: int = 1500,
    n_features: int = 12,
    n_informative: int = 6,
    n_redundant: int = 2,
    n_clusters_per_class: int = 1,
    class_sep: float = 1.2,
    flip_y: float = 0.02,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a clean synthetic binary classification DataFrame."""
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=n_redundant,
        n_clusters_per_class=n_clusters_per_class,
        class_sep=class_sep,
        flip_y=flip_y,
        random_state=seed,
    )
    feature_names = [f"feat_{i:02d}" for i in range(n_features)]
    df = pd.DataFrame(X, columns=feature_names)
    df["target"] = y.astype(int)
    return df


def load_breast_cancer_df() -> pd.DataFrame:
    """Load the scikit-learn bundled breast cancer dataset as a DataFrame."""
    data = load_breast_cancer(as_frame=True)
    df = data.frame.copy()
    # Ensure clean column names without spaces or parentheses
    df.columns = [
        col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")
        for col in df.columns
    ]
    return df


def register_builtin_datasets() -> None:
    """Register standard built-in datasets in the registry."""
    register_dataset(
        DatasetSpec(
            name="synthetic_easy",
            load=lambda: make_synthetic_dataset(
                n_samples=1500,
                n_features=10,
                n_informative=6,
                class_sep=1.5,
                flip_y=0.08,
                seed=101,
            ),
            target="target",
            positive="1",
            role="dev",
            description="Easy synthetic classification problem with high class separation.",
        )
    )

    register_dataset(
        DatasetSpec(
            name="synthetic_medium",
            load=lambda: make_synthetic_dataset(
                n_samples=1500,
                n_features=12,
                n_informative=6,
                class_sep=1.0,
                flip_y=0.10,
                seed=102,
            ),
            target="target",
            positive="1",
            role="dev",
            description="Medium synthetic classification problem with moderate noise.",
        )
    )

    register_dataset(
        DatasetSpec(
            name="synthetic_hard",
            load=lambda: make_synthetic_dataset(
                n_samples=1500,
                n_features=14,
                n_informative=6,
                class_sep=0.8,
                flip_y=0.12,
                seed=103,
            ),
            target="target",
            positive="1",
            role="heldout",
            description="Hard synthetic classification problem with lower separation.",
        )
    )

    register_dataset(
        DatasetSpec(
            name="sklearn_breast_cancer",
            load=load_breast_cancer_df,
            target="target",
            positive="1",
            role="dev",
            description="Scikit-learn bundled breast cancer diagnostic dataset (n=569).",
        )
    )

    register_dataset(
        DatasetSpec(
            name="synth_large",
            load=lambda: make_synthetic_dataset(
                n_samples=3000,
                n_features=15,
                n_informative=8,
                class_sep=1.0,
                seed=104,
            ),
            target="target",
            positive="1",
            role="dev",
            description="Large synthetic binary classification problem (n=3000, 15 features).",
        )
    )
