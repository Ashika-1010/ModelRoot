"""Assessment package: metrics, baseline evaluation, reference panel, verdict (M5/M6)."""

from __future__ import annotations

from modelroot.assessment.baseline import BaselineRun, PredictionSet, dummy_baseline, fit_baseline
from modelroot.assessment.metrics import (
    LABEL_METRICS,
    METRIC_NAMES,
    PROBA_METRICS,
    UndefinedMetric,
    compute_metrics,
    metric_fn,
    needs_proba,
)
from modelroot.assessment.reference_panel import (
    PanelEntry,
    PanelResult,
    fit_reference_panel,
)
from modelroot.assessment.verdict import assess, assess_only, get_metric_value

__all__ = [
    "LABEL_METRICS",
    "METRIC_NAMES",
    "PROBA_METRICS",
    "BaselineRun",
    "PanelEntry",
    "PanelResult",
    "PredictionSet",
    "UndefinedMetric",
    "assess",
    "assess_only",
    "compute_metrics",
    "dummy_baseline",
    "fit_baseline",
    "fit_reference_panel",
    "get_metric_value",
    "metric_fn",
    "needs_proba",
]
