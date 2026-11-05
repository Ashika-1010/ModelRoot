"""High-level pipeline entry points (M6/M9).

``assess_only`` is the M6 entry point.
``diagnose`` is the M9 entry point that runs the full hypothesis engine.
``_prepare`` extracts the shared data-loading + baseline step used by both.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime

from modelroot.assessment.baseline import BaselineRun, fit_baseline
from modelroot.assessment.reference_panel import _get_col_types, fit_reference_panel
from modelroot.assessment.verdict import assess, assess_only
from modelroot.config.thresholds import Thresholds
from modelroot.data.fingerprint import dataset_fingerprint
from modelroot.data.loader import Limits, load_csv
from modelroot.data.split import SplitData, make_split
from modelroot.data.validate import ValidatedDataset, validate_for_analysis
from modelroot.experiments.budget import BudgetExceeded, BudgetTracker
from modelroot.experiments.cache import PredictionCache
from modelroot.experiments.runner import ExperimentFailure, ExperimentRunner
from modelroot.hypotheses.base import Applicability, Hypothesis, HypothesisContext
from modelroot.hypotheses.registry import get_registry
from modelroot.models.registry import get_model
from modelroot.schemas.config import AnalysisConfig
from modelroot.schemas.experiment import ExperimentResult, ExperimentSpec
from modelroot.schemas.hypothesis import HypothesisResult
from modelroot.schemas.result import AnalysisResult, AssessmentResult
from modelroot.utils.seeds import SeedManager

# ---------------------------------------------------------------------------
# Re-export for M6 compatibility
# ---------------------------------------------------------------------------

__all__ = ["_prepare", "assess_only", "diagnose"]


# ---------------------------------------------------------------------------
# PreparedAnalysis: shared result of the load→validate→split→baseline step
# ---------------------------------------------------------------------------


@dataclass
class PreparedAnalysis:
    """Intermediate result from ``_prepare``.

    Attributes:
        config: Original analysis configuration.
        validated: Validated dataset.
        split: Data split.
        col_types: Column type mapping.
        baseline_run: Fitted baseline model and predictions.
        dataset_fp: Dataset fingerprint.
        split_fp: Split fingerprint.
        sm: Seed manager (already used for split and baseline seeds).
    """

    config: AnalysisConfig
    validated: ValidatedDataset
    split: SplitData
    col_types: dict[str, str]
    baseline_run: BaselineRun
    dataset_fp: str
    split_fp: str
    sm: SeedManager


def _prepare(
    config: AnalysisConfig,
    limits: Limits | None = None,
) -> PreparedAnalysis:
    """Load, validate, split, and fit the baseline model.

    This is the shared first phase of both ``assess_only`` and ``diagnose``.
    Extracting it avoids code duplication while preserving ``assess_only``
    behaviour exactly.

    Args:
        config: Analysis configuration.
        limits: Resource limits for data loading.

    Returns:
        :class:`PreparedAnalysis` with all artifacts ready.

    Raises:
        ContractViolation: If data loading or validation fails.
        InvalidModelSpec: If the model spec is invalid.
    """
    if limits is None:
        limits = Limits()

    df = load_csv(config.data_path, limits=limits)
    validated: ValidatedDataset = validate_for_analysis(df, config)

    sm = SeedManager(config.seed)
    split_seed = sm.spawn("split")
    split = make_split(validated, config, seed=split_seed, limits=limits)

    col_types = _get_col_types(split)
    adapter = get_model(config.model_spec, col_types, seed=sm.spawn("baseline"))
    baseline_run = fit_baseline(adapter, split, primary_metric=config.primary_metric.value)

    dataset_fp = dataset_fingerprint(validated.df)
    split_fp = split.split_fingerprint

    return PreparedAnalysis(
        config=config,
        validated=validated,
        split=split,
        col_types=col_types,
        baseline_run=baseline_run,
        dataset_fp=dataset_fp,
        split_fp=split_fp,
        sm=sm,
    )


# ---------------------------------------------------------------------------
# diagnose — M9 entry point
# ---------------------------------------------------------------------------


def diagnose(
    config: AnalysisConfig,
    hypotheses: list[type[Hypothesis]] | None = None,
    thresholds: Thresholds | None = None,
    on_progress: Callable[[str, float], None] | None = None,
) -> AnalysisResult:
    """Run the full analysis pipeline with the hypothesis engine.

    Pipeline stages:
      1. Load → validate → split → fit baseline (via ``_prepare``).
      2. Assessment verdict (reference panel + 4 rules).
      3. For each applicable, triggered hypothesis: plan → run experiments
         → evaluate → collect results.
      4. Build and return ``AnalysisResult``.

    Args:
        config: Analysis configuration.
        hypotheses: List of hypothesis *classes* to evaluate.  Defaults to
            the global registry (currently empty until M11).
        thresholds: Decision thresholds.  Defaults to ``Thresholds()``.
        on_progress: Optional callback ``(stage: str, fraction: float)``.

    Returns:
        A completed :class:`AnalysisResult`.

    Notes:
        Synchronous operations cannot be forcibly interrupted.  A budget
        overrun is bounded by the duration of the single operation that
        exceeded the limit.
    """
    if thresholds is None:
        thresholds = Thresholds()

    if hypotheses is None:
        hypotheses = get_registry()

    t_start = time.monotonic()
    timings: dict[str, float] = {}

    def _progress(stage: str, frac: float) -> None:
        if on_progress is not None:
            on_progress(stage, frac)

    _progress("prepare", 0.0)
    t0 = time.monotonic()
    prep = _prepare(config)
    timings["prepare"] = time.monotonic() - t0

    _progress("assess", 0.2)
    t0 = time.monotonic()
    from modelroot.assessment.baseline import dummy_baseline

    panel_result = fit_reference_panel(
        split=prep.split,
        primary_metric=config.primary_metric.value,
        base_seed=config.seed,
    )
    dummy_metrics = dummy_baseline(prep.split)
    assessment_partial = assess(
        config=config,
        split=prep.split,
        baseline_run=prep.baseline_run,
        panel_result=panel_result,
        thresholds=thresholds,
    )
    assessment_result = AssessmentResult(
        baseline_metrics=assessment_partial.baseline_metrics,
        dummy_metrics=dummy_metrics,
        reference=assessment_partial.reference,
        headroom=assessment_partial.headroom,
        train_holdout_gap=assessment_partial.train_holdout_gap,
        current_drop=assessment_partial.current_drop,
        underperforming=assessment_partial.underperforming,
        reasons=assessment_partial.reasons,
        headline=assessment_partial.headline,
    )
    timings["assess"] = time.monotonic() - t0

    # Build context for hypothesis evaluation
    ctx = HypothesisContext(
        split=prep.split,
        baseline_run=prep.baseline_run,
        config=config,
        col_types=prep.col_types,
        seeds=prep.sm,
    )

    budget = BudgetTracker(config.budget_s)
    cache = PredictionCache()
    runner = ExperimentRunner(ctx=ctx, cache=cache, budget=budget)

    all_hyp_results: list[HypothesisResult] = []
    all_experiments: list[ExperimentResult] = []
    all_specs: list[ExperimentSpec] = []
    warnings: list[str] = []

    n_hypotheses = len(hypotheses)
    for h_idx, hyp_cls in enumerate(hypotheses):
        _progress("hypotheses", 0.4 + 0.5 * h_idx / max(n_hypotheses, 1))

        try:
            hyp = hyp_cls()

            # Check budget before each hypothesis
            try:
                budget.check()
            except BudgetExceeded as exc:
                warnings.append(f"Budget exhausted before {hyp_cls.__name__}: {exc}")
                break

            # Applicability
            applicability: Applicability = hyp.applicable(ctx)
            if not applicability.applicable:
                continue

            # Trigger
            trigger = hyp.trigger(ctx)

            # Plan + run experiments (only if trigger fired and run_experiments=True)
            exp_results: list[ExperimentResult] = []
            if trigger.fired and config.run_experiments:
                specs = hyp.plan(ctx, trigger)
                for spec in specs:
                    all_specs.append(spec)
                    outcome = runner.run(spec)
                    if isinstance(outcome, ExperimentFailure):
                        status_label = outcome.status.value
                        warnings.append(f"Experiment {spec.id} [{status_label}]: {outcome.message}")
                        # Represent as a failed ExperimentResult (FW-ERR/FW-TMO)
                        failed_result = ExperimentResult(
                            spec_id=spec.id,
                            status=outcome.status,
                            baseline_metric=0.0,
                            intervention_metric=0.0,
                            delta=0.0,
                            ci_low=0.0,
                            ci_high=0.0,
                            n_eval=0,
                            n_valid_boot=0,
                            runtime_s=outcome.runtime_s,
                            error=outcome.message,
                        )
                        all_experiments.append(failed_result)
                    else:
                        all_experiments.append(outcome)
                        exp_results.append(outcome)

            # Evaluate
            hyp_results = hyp.evaluate(ctx, trigger, exp_results)
            all_hyp_results.extend(hyp_results)

        except BudgetExceeded as exc:
            warnings.append(f"Budget exceeded during {hyp_cls.__name__}: {exc}")
            break
        except Exception as exc:
            warnings.append(f"Hypothesis {hyp_cls.__name__} failed: {type(exc).__name__}: {exc}")

    timings["hypotheses"] = time.monotonic() - t0
    timings["total"] = time.monotonic() - t_start

    _progress("finalize", 0.95)

    # M10 finalization: rank, assign severity, validate
    from modelroot.evidence.rules import RULE_SET_VERSION, finalize

    all_hyp_results = finalize(all_hyp_results, thresholds)

    import modelroot

    versions = {
        "modelroot": modelroot.__version__,
    }

    # reproducibility_hash
    import hashlib

    repr_payload = f"{prep.dataset_fp}|{prep.split_fp}|{config.seed}|{config.primary_metric.value}"
    reproducibility_hash = hashlib.sha256(repr_payload.encode()).hexdigest()

    result = AnalysisResult(
        analysis_id=str(uuid.uuid4()),
        created_at=datetime.now(tz=UTC),
        status="completed",
        config=config,
        dataset_fingerprint=prep.dataset_fp,
        split_fingerprint=prep.split_fp,
        versions=versions,
        rule_set_version=RULE_SET_VERSION,
        thresholds={
            "headroom": thresholds.headroom,
            "gap": thresholds.gap,
            "current_drop": thresholds.current_drop,
            "delta_min": thresholds.delta_min,
            "alpha": thresholds.alpha,
        },
        assessment=assessment_result,
        hypotheses=all_hyp_results,
        experiments=all_experiments,
        evidence=[],
        warnings=warnings,
        timings=timings,
        reproducibility_hash=reproducibility_hash,
    )

    _progress("done", 1.0)
    return result
