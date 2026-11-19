"""Dataset registry and DatasetSpec contract (M13).

Provides registration and lookup of benchmark datasets.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import pandas as pd

Role = Literal["dev", "heldout"]


@dataclass(frozen=True)
class DatasetSpec:
    """Specification of a benchmark dataset.

    Attributes:
        name: Unique dataset identifier.
        load: Zero-arg callable returning the raw pandas DataFrame.
        target: Target column name.
        positive: String representation of the positive class label.
        role: Development role ('dev' for threshold tuning, 'heldout' for evaluation).
        description: Human-readable dataset description.
    """

    name: str
    load: Callable[[], pd.DataFrame]
    target: str
    positive: str = "1"
    role: Role = "dev"
    description: str = ""


_DATASETS: dict[str, DatasetSpec] = {}


def register_dataset(spec: DatasetSpec) -> DatasetSpec:
    """Register a DatasetSpec in the global benchmark registry.

    Args:
        spec: The dataset specification.

    Returns:
        The registered DatasetSpec.

    Raises:
        ValueError: If a dataset with the same name is already registered.
    """
    if spec.name in _DATASETS:
        raise ValueError(f"Dataset {spec.name!r} is already registered.")
    _DATASETS[spec.name] = spec
    return spec


def get_dataset(name: str) -> DatasetSpec:
    """Retrieve a registered DatasetSpec by name.

    Args:
        name: Dataset identifier.

    Returns:
        The registered DatasetSpec.

    Raises:
        KeyError: If *name* is not registered.
    """
    # Ensure built-ins are registered
    _ensure_builtins()
    if name not in _DATASETS:
        raise KeyError(f"Dataset {name!r} is not registered. Available: {sorted(_DATASETS.keys())}")
    return _DATASETS[name]


def list_datasets() -> list[str]:
    """Return a sorted list of all registered dataset names."""
    _ensure_builtins()
    return sorted(_DATASETS.keys())


def clear_dataset_registry() -> None:
    """Clear all registered datasets (testing utility only)."""
    _DATASETS.clear()


_BUILTINS_REGISTERED = False


def _ensure_builtins() -> None:
    """Ensure standard built-in offline datasets are registered."""
    global _BUILTINS_REGISTERED
    if _BUILTINS_REGISTERED:
        return
    _BUILTINS_REGISTERED = True

    from modelroot_bench.datasets.synthetic import register_builtin_datasets

    register_builtin_datasets()
