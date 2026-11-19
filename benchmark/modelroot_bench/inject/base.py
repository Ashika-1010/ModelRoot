"""Base injector interface and InjectedScenario contract (M13)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.scenario import GroundTruth


@dataclass
class InjectedScenario:
    """The result of applying a failure injector to a base dataset.

    Attributes:
        df: The modified DataFrame with features, target, and optional split column.
        model_spec: Algorithm name and hyperparameters to evaluate.
        ground_truth: GroundTruth contract describing what was injected.
        clean_df: Uncorrupted reference DataFrame for oracle impact comparison.
    """

    df: pd.DataFrame
    model_spec: dict[str, Any]
    ground_truth: GroundTruth
    clean_df: pd.DataFrame


class Injector(ABC):
    """Abstract protocol for failure injectors."""

    @abstractmethod
    def apply(
        self,
        df: pd.DataFrame,
        dataset_spec: DatasetSpec,
        severity: str,
        seed: int,
        model_spec: dict[str, Any] | None = None,
        params: dict[str, Any] | None = None,
    ) -> InjectedScenario:
        """Apply the failure injection.

        Args:
            df: Base dataset DataFrame.
            dataset_spec: Specification of the base dataset.
            severity: Injected severity string (e.g., 'S1', 'S2', 'S3').
            seed: Deterministic random seed.
            model_spec: Optional base model specification.
            params: Optional additional injector parameters.

        Returns:
            An :class:`InjectedScenario`.
        """
        ...
