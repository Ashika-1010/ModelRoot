"""Experiment runner (M9).

``ExperimentRunner.run`` executes one ``ExperimentSpec`` against the current
context, producing either an ``ExperimentResult`` or a failure record.

Failed experiments must NOT be emitted as completed ``ExperimentResult``
objects.  Instead:
- ``BudgetExceeded`` → ``status=timeout``, a notes entry, and a warning.
- Any other exception → ``status=error``, an error field, a notes entry,
  and a warning.
Failures are never cached.
"""

from __future__ import annotations

import hashlib
import time
from dataclasses import dataclass

from modelroot.assessment.metrics import needs_proba
from modelroot.experiments.budget import BudgetExceeded, BudgetTracker
from modelroot.experiments.cache import PredictionCache
from modelroot.schemas.common import ExperimentStatus
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec


@dataclass
class ExperimentFailure:
    """Represents a failed experiment (timeout or error).

    Attributes:
        spec_id: References the ``ExperimentSpec.id``.
        status: ``timeout`` or ``error``.
        message: Human-readable failure message.
        runtime_s: Elapsed time before failure.
    """

    spec_id: str
    status: ExperimentStatus
    message: str
    runtime_s: float


def _spec_hash(spec: ExperimentSpec) -> str:
    """Compute a stable hash of an ``ExperimentSpec`` for cache keying.

    Args:
        spec: The experiment specification.

    Returns:
        16-character hex digest.
    """
    payload = (
        f"{spec.intervention}|{sorted(spec.params.items())}|"
        f"{spec.metric}|{spec.eval_on}|{spec.n_boot}|{spec.repeats}|{spec.seed}"
    )
    return hashlib.sha256(payload.encode()).hexdigest()[:16]


class ExperimentRunner:
    """Runs experiment specifications against a fixed analysis context.

    Args:
        ctx: :class:`~modelroot.hypotheses.base.HypothesisContext`.
        cache: Shared prediction cache.
        budget: Parent budget tracker.
    """

    def __init__(
        self,
        ctx: object,
        cache: PredictionCache,
        budget: BudgetTracker,
    ) -> None:
        self._ctx = ctx
        self._cache = cache
        self._budget = budget

    def run(self, spec: ExperimentSpec) -> ExperimentResult | ExperimentFailure:
        """Execute *spec* and return either a result or a failure record.

        Failures (timeout or error) are never cached.

        Args:
            spec: The experiment to run.

        Returns:
            An :class:`ExperimentResult` on success, or an
            :class:`ExperimentFailure` on timeout / error.
        """
        t0 = time.monotonic()

        try:
            # Cooperative budget check before the experiment
            self._budget.check()

            result = self._execute(spec, t0)
            return result

        except BudgetExceeded:
            runtime_s = time.monotonic() - t0
            return ExperimentFailure(
                spec_id=spec.id,
                status=ExperimentStatus.timeout,
                message="Budget exceeded before or during experiment.",
                runtime_s=runtime_s,
            )
        except Exception as exc:
            runtime_s = time.monotonic() - t0
            return ExperimentFailure(
                spec_id=spec.id,
                status=ExperimentStatus.error,
                message=str(exc),
                runtime_s=runtime_s,
            )

    def _execute(
        self,
        spec: ExperimentSpec,
        t0: float,
    ) -> ExperimentResult:
        """Run the experiment and return a completed ``ExperimentResult``.

        Raises any exception on failure — caller wraps in try/except.

        Args:
            spec: Experiment specification.
            t0: Start time for elapsed measurement.

        Returns:
            :class:`ExperimentResult` with status ``completed``.
        """
        from modelroot.experiments.interventions import get_intervention
        from modelroot.stats.bootstrap import paired_bootstrap

        ctx = self._ctx
        baseline_run = ctx.baseline_run  # type: ignore[attr-defined]

        # Retrieve or build intervention outcome
        intervention_fn = get_intervention(spec.intervention)
        outcome = intervention_fn(ctx, spec.params)

        # Baseline predictions (use train split to get holdout baseline)
        if "holdout" in spec.eval_on:
            eval_y_true = baseline_run.holdout_pred.y_true
            eval_y_pred_base = baseline_run.holdout_pred.y_pred
            eval_y_proba_base = baseline_run.holdout_pred.y_proba
            eval_y_pred_int = outcome.holdout_pred
            eval_y_proba_int = outcome.holdout_proba
            n_eval = len(eval_y_true)
        else:
            # current fallback
            if baseline_run.current_pred is None:
                raise ValueError("Experiment requires 'current' eval_on but no current data.")
            eval_y_true = baseline_run.current_pred.y_true
            eval_y_pred_base = baseline_run.current_pred.y_pred
            eval_y_proba_base = baseline_run.current_pred.y_proba
            eval_y_pred_int = outcome.current_pred
            eval_y_proba_int = outcome.current_proba
            n_eval = len(eval_y_true)

        use_proba = needs_proba(spec.metric)
        y_proba_a = eval_y_proba_base if use_proba else None
        y_proba_b = eval_y_proba_int if use_proba else None

        # Paired bootstrap
        boot_result = paired_bootstrap(
            y_true=eval_y_true,
            pred_a=eval_y_pred_base,
            pred_b=eval_y_pred_int,
            metric=spec.metric,
            n_boot=spec.n_boot,
            seed=spec.seed,
            y_proba_a=y_proba_a,
            y_proba_b=y_proba_b,
        )

        runtime_s = time.monotonic() - t0

        return ExperimentResult(
            spec_id=spec.id,
            status=ExperimentStatus.completed,
            baseline_metric=boot_result.metric_a,
            intervention_metric=boot_result.metric_b,
            delta=boot_result.delta,
            ci_low=boot_result.ci_low,
            ci_high=boot_result.ci_high,
            n_eval=n_eval,
            n_valid_boot=boot_result.n_valid,
            runtime_s=runtime_s,
        )
