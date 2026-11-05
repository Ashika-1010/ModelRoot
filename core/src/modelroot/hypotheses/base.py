"""Hypothesis base class and supporting types (M9)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from modelroot.schemas.common import HypothesisType
from modelroot.schemas.hypothesis import HypothesisResult, TriggerResult


@dataclass
class HypothesisContext:
    """Context passed to each hypothesis during an analysis run.

    Attributes:
        split: The data split.
        baseline_run: The fitted baseline run.
        config: The analysis configuration.
        col_types: Column type mapping for model instantiation.
        seeds: Seed manager for reproducible randomness.
    """

    split: Any  # SplitData
    baseline_run: Any  # BaselineRun
    config: Any  # AnalysisConfig
    col_types: dict[str, str]
    seeds: Any  # SeedManager


@dataclass(frozen=True)
class Applicability:
    """Whether a hypothesis is applicable to this analysis.

    Attributes:
        applicable: ``True`` if the hypothesis should run.
        reason: Why it is not applicable (if ``applicable`` is ``False``).
    """

    applicable: bool
    reason: str = ""


class Hypothesis(ABC):
    """Abstract base class for all ModelRoot hypotheses.

    Every concrete hypothesis must implement ``applicable``, ``trigger``,
    ``plan`` and ``evaluate``.  The ``detector_flag`` method is used in
    detector-only ablation mode.
    """

    type: HypothesisType

    @abstractmethod
    def applicable(self, ctx: HypothesisContext) -> Applicability:
        """Return whether this hypothesis should run for the given context.

        Args:
            ctx: Current analysis context.

        Returns:
            :class:`Applicability` with flag and optional reason.
        """

    @abstractmethod
    def trigger(self, ctx: HypothesisContext) -> TriggerResult:
        """Run the cheap trigger check and return diagnostic evidence.

        Args:
            ctx: Current analysis context.

        Returns:
            :class:`TriggerResult` indicating whether the trigger fired.
        """

    @abstractmethod
    def plan(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
    ) -> list[Any]:  # list[ExperimentSpec]
        """Return the list of experiments to run given the trigger result.

        Args:
            ctx: Current analysis context.
            trigger: Result from the trigger check.

        Returns:
            List of :class:`ExperimentSpec` objects.
        """

    @abstractmethod
    def evaluate(
        self,
        ctx: HypothesisContext,
        trigger: TriggerResult,
        results: list[Any],  # list[ExperimentResult]
    ) -> list[HypothesisResult]:
        """Produce verdict(s) from trigger and experiment results.

        Args:
            ctx: Current analysis context.
            trigger: Result from the trigger check.
            results: List of experiment results (may be empty).

        Returns:
            List of :class:`HypothesisResult` objects (one per sub-claim).
        """

    def detector_flag(self, trigger: TriggerResult) -> bool:
        """Return whether this hypothesis should be flagged in detector-only mode.

        Defaults to ``trigger.fired``; hypotheses may override for finer control.

        Args:
            trigger: Result from the trigger check.

        Returns:
            ``True`` if the hypothesis would be flagged.
        """
        return trigger.fired
