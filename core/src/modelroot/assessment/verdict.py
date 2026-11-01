"""Assessment verdict logic (M6).

``assess`` evaluates whether the model is underperforming by applying four
rules.  ``assess_only`` is the high-level entry point that accepts an
``AnalysisConfig`` and runs the full pipeline through the verdict.
"""

from __future__ import annotations

from modelroot.assessment.baseline import BaselineRun, fit_baseline
from modelroot.assessment.reference_panel import PanelResult, _get_col_types, fit_reference_panel
from modelroot.config.thresholds import Thresholds
from modelroot.data.loader import Limits, load_csv
from modelroot.data.split import SplitData, make_split
from modelroot.data.validate import ValidatedDataset, validate_for_analysis
from modelroot.models.registry import get_model
from modelroot.schemas.config import AnalysisConfig
from modelroot.schemas.metrics import MetricResult
from modelroot.schemas.result import AssessmentResult

# Small epsilon for floating-point boundary comparisons
_EPS = 1e-9


# ---------------------------------------------------------------------------
# Public helpers
# ---------------------------------------------------------------------------


def get_metric_value(metrics: list[MetricResult], name: str, split: str) -> float | None:
    """Extract a metric value from a list of MetricResult objects.

    Args:
        metrics: List of ``MetricResult`` objects.
        name: Metric name to look up.
        split: Split label to match.

    Returns:
        The float value, or ``None`` if not found.
    """
    for m in metrics:
        if m.name == name and m.split == split:
            return float(m.value)
    return None


# ---------------------------------------------------------------------------
# assess
# ---------------------------------------------------------------------------


def assess(
    config: AnalysisConfig,
    split: SplitData,
    baseline_run: BaselineRun,
    panel_result: PanelResult,
    thresholds: Thresholds,
) -> AssessmentResult:
    """Produce an ``AssessmentResult`` from pre-computed baseline and panel.

    Four underperformance rules are applied in fixed order:
      1. ``reference - baseline >= thresholds.headroom`` on primary metric.
      2. ``baseline_holdout < config.metadata.expected_score`` (if set).
      3. ``train - holdout >= thresholds.gap`` on primary metric.
      4. ``holdout - current >= thresholds.current_drop`` (if labeled current).

    Args:
        config: Analysis configuration.
        split: The data split used.
        baseline_run: Fitted baseline predictions and metrics.
        panel_result: Reference panel result.
        thresholds: Decision thresholds.

    Returns:
        A completed ``AssessmentResult``.
    """
    pm = config.primary_metric.value

    baseline_holdout = get_metric_value(baseline_run.metrics, pm, "holdout") or 0.0
    baseline_train = get_metric_value(baseline_run.metrics, pm, "train") or 0.0
    for m in panel_result.reference.metrics:
        if m.name == pm:
            ref_holdout = float(m.value)
            break

    headroom = ref_holdout - baseline_holdout
    train_holdout_gap = baseline_train - baseline_holdout

    current_drop: float | None = None
    if split.y_current is not None and baseline_run.current_pred is not None:
        baseline_current = get_metric_value(baseline_run.metrics, pm, "current")
        if baseline_current is not None:
            current_drop = baseline_holdout - baseline_current

    # Evaluate rules
    reasons: list[str] = []
    underperforming = False

    # Rule 1: headroom
    if headroom >= thresholds.headroom - _EPS:
        reasons.append(
            f"Reference model outperforms baseline by {headroom:.4f} "
            f"(threshold {thresholds.headroom})."
        )
        underperforming = True

    # Rule 2: expected score
    if (
        config.metadata.expected_score is not None
        and baseline_holdout < config.metadata.expected_score - _EPS
    ):
        reasons.append(
            f"Holdout {pm}={baseline_holdout:.4f} is below expected "
            f"score {config.metadata.expected_score:.4f}."
        )
        underperforming = True

    # Rule 3: train-holdout gap
    if train_holdout_gap >= thresholds.gap - _EPS:
        reasons.append(
            f"Train-holdout gap is {train_holdout_gap:.4f} (threshold {thresholds.gap})."
        )
        underperforming = True

    # Rule 4: current drop
    if current_drop is not None and current_drop >= thresholds.current_drop - _EPS:
        reasons.append(
            f"Performance dropped {current_drop:.4f} on current data "
            f"(threshold {thresholds.current_drop})."
        )
        underperforming = True

    headline = (
        f"Baseline {pm}={baseline_holdout:.4f}, "
        f"reference {pm}={ref_holdout:.4f}. "
        f"{'Underperforming.' if underperforming else 'Performing adequately.'}"
    )

    return AssessmentResult(
        baseline_metrics=baseline_run.metrics,
        dummy_metrics=[],  # populated by assess_only
        reference=panel_result.reference,
        headroom=headroom,
        train_holdout_gap=train_holdout_gap,
        current_drop=current_drop,
        underperforming=underperforming,
        reasons=reasons,
        headline=headline,
    )


# ---------------------------------------------------------------------------
# assess_only — high-level pipeline entry point
# ---------------------------------------------------------------------------


def assess_only(
    config: AnalysisConfig,
    thresholds: Thresholds | None = None,
) -> AssessmentResult:
    """Run the full load → validate → split → baseline → panel → assess pipeline.

    Args:
        config: Complete analysis configuration.
        thresholds: Decision thresholds; defaults to ``Thresholds()``.

    Returns:
        A completed ``AssessmentResult``.

    Raises:
        ContractViolation: If any data loading or validation step fails.
        InvalidModelSpec: If the model spec is invalid.
    """
    from modelroot.assessment.baseline import dummy_baseline

    if thresholds is None:
        thresholds = Thresholds()

    limits = Limits()
    df = load_csv(config.data_path, limits=limits)
    validated: ValidatedDataset = validate_for_analysis(df, config)

    from modelroot.utils.seeds import SeedManager

    sm = SeedManager(config.seed)
    split_seed = sm.spawn("split")
    split = make_split(validated, config, seed=split_seed, limits=limits)

    col_types = _get_col_types(split)
    adapter = get_model(config.model_spec, col_types, seed=sm.spawn("baseline"))
    baseline_run = fit_baseline(adapter, split, primary_metric=config.primary_metric.value)

    panel_result = fit_reference_panel(
        split=split,
        primary_metric=config.primary_metric.value,
        base_seed=config.seed,
    )

    dummy_metrics = dummy_baseline(split)

    result = assess(
        config=config,
        split=split,
        baseline_run=baseline_run,
        panel_result=panel_result,
        thresholds=thresholds,
    )

    # Attach dummy metrics (assess() leaves them empty)
    return AssessmentResult(
        baseline_metrics=result.baseline_metrics,
        dummy_metrics=dummy_metrics,
        reference=result.reference,
        headroom=result.headroom,
        train_holdout_gap=result.train_holdout_gap,
        current_drop=result.current_drop,
        underperforming=result.underperforming,
        reasons=result.reasons,
        headline=result.headline,
    )
