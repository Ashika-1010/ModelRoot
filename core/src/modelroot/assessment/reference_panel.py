"""Reference panel evaluation (M6).

The reference panel fits Logistic Regression, Random Forest and
HistGradientBoosting at their M4 defaults and reports holdout scores.
The best result (highest primary-metric score) is selected as the
reference for the assessment verdict.  Ties are broken by stable
algorithm ordering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from modelroot.assessment.metrics import _METRIC_REGISTRY, UndefinedMetric, _validate_inputs
from modelroot.data.split import SplitData
from modelroot.models.registry import get_model
from modelroot.schemas.common import Algorithm
from modelroot.schemas.config import ModelSpec
from modelroot.schemas.metrics import MetricResult
from modelroot.schemas.result import ReferenceResult

# Panel algorithms in stable order (LR, RF, HGB — DT excluded per spec)
_PANEL_ALGORITHMS: tuple[Algorithm, ...] = (
    Algorithm.logistic_regression,
    Algorithm.random_forest,
    Algorithm.hist_gradient_boosting,
)

# Deterministic seed offset per panel position to ensure reproducibility
_PANEL_SEED_OFFSET = 1000


@dataclass
class PanelEntry:
    """Result for a single panel algorithm.

    Attributes:
        algorithm: The algorithm used.
        holdout_score: Primary-metric score on the holdout set.
        metrics: All holdout metrics.
        seed: The seed used to train this adapter.
    """

    algorithm: Algorithm
    holdout_score: float
    metrics: list[MetricResult]
    seed: int


@dataclass
class PanelResult:
    """Full reference panel result.

    Attributes:
        entries: One entry per panel algorithm, in training order.
        best: Index into *entries* of the best-scoring algorithm.
        reference: The :class:`ReferenceResult` schema object for the verdict.
    """

    entries: list[PanelEntry]
    best: int
    reference: ReferenceResult


def fit_reference_panel(
    split: SplitData,
    primary_metric: str,
    base_seed: int,
) -> PanelResult:
    """Fit the three reference algorithms and return a :class:`PanelResult`.

    Each algorithm is fitted with its M4 defaults.  Seeds are derived
    deterministically from *base_seed* + per-algorithm offset so results
    are reproducible.  Errors during fitting or prediction propagate (not
    silently skipped).

    Args:
        split: Train/holdout/current split data.
        primary_metric: Name of the primary metric used to select the best
            reference model.
        base_seed: Master seed from the analysis config.

    Returns:
        :class:`PanelResult` with the best reference result selected.

    Raises:
        Exception: If any panel model fails to fit or predict.
        UndefinedMetric: If the primary metric is undefined for the panel data.
    """
    entries: list[PanelEntry] = []

    for i, algo in enumerate(_PANEL_ALGORITHMS):
        seed = base_seed + _PANEL_SEED_OFFSET + i
        spec = ModelSpec(algorithm=algo, params={})
        adapter = get_model(spec, _get_col_types(split), seed=seed)
        adapter.fit(split.X_train, split.y_train)

        raw_proba_2d = adapter.predict_proba(split.X_holdout)
        if raw_proba_2d.ndim != 2 or raw_proba_2d.shape[1] != 2:
            raise ValueError(f"[{algo.value}] predict_proba returned shape {raw_proba_2d.shape}")
        y_pred = adapter.predict(split.X_holdout)
        y_proba = raw_proba_2d[:, 1]
        y_true = np.asarray(split.y_holdout.values, dtype=int)

        y_true_v, y_pred_v, y_proba_v = _validate_inputs(y_true, y_pred, y_proba)

        panel_metrics: list[MetricResult] = []
        holdout_score = 0.0
        for name, fn in _METRIC_REGISTRY.items():
            try:
                val = fn(y_true_v, y_pred_v, y_proba_v)
                panel_metrics.append(
                    MetricResult(name=name, value=val, split="holdout", n=len(y_true_v))
                )
                if name == primary_metric:
                    holdout_score = val
            except UndefinedMetric:
                pass

        # If primary metric was never computed (e.g. undefined), try to get it
        found = any(m.name == primary_metric for m in panel_metrics)
        if not found:
            raise UndefinedMetric(
                f"Primary metric '{primary_metric}' is undefined for panel algorithm {algo.value}."
            )

        entries.append(
            PanelEntry(
                algorithm=algo,
                holdout_score=holdout_score,
                metrics=panel_metrics,
                seed=seed,
            )
        )

    # Select best by primary metric; tie-break by stable _PANEL_ALGORITHMS order
    best_idx = int(np.argmax([e.holdout_score for e in entries]))

    best_entry = entries[best_idx]
    reference = ReferenceResult(
        algorithm=best_entry.algorithm,
        metrics=best_entry.metrics,
    )

    return PanelResult(entries=entries, best=best_idx, reference=reference)


def _get_col_types(split: SplitData) -> dict[str, str]:
    """Infer column types from the training DataFrame.

    Falls back to 'categorical' for any column that is not clearly numeric.

    Args:
        split: SplitData containing ``X_train``.

    Returns:
        Dict mapping column name to ``"numeric"`` or ``"categorical"``.
    """
    import pandas as pd

    col_types: dict[str, str] = {}
    for col in split.X_train.columns:
        series = split.X_train[col].replace("", pd.NA)
        non_null = series.dropna()
        if len(non_null) == 0:
            col_types[col] = "categorical"
            continue
        try:
            pd.to_numeric(non_null)
            col_types[col] = "numeric"
        except (ValueError, TypeError):
            col_types[col] = "categorical"
    return col_types
