"""Tests for M5 metric computation (T5-01 to T5-09)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    roc_auc_score,
)

from modelroot.assessment.metrics import (
    LABEL_METRICS,
    METRIC_NAMES,
    PROBA_METRICS,
    UndefinedMetric,
    compute_metrics,
    metric_fn,
    needs_proba,
)

# ---------------------------------------------------------------------------
# T5-01: Constants
# ---------------------------------------------------------------------------


def test_metric_names_order() -> None:
    assert METRIC_NAMES == (
        "accuracy",
        "balanced_accuracy",
        "macro_f1",
        "f1",
        "precision",
        "recall",
        "roc_auc",
        "pr_auc",
    )


def test_label_metrics_set() -> None:
    assert "accuracy" in LABEL_METRICS
    assert "roc_auc" not in LABEL_METRICS


def test_proba_metrics_set() -> None:
    assert "roc_auc" in PROBA_METRICS
    assert "pr_auc" in PROBA_METRICS
    assert "accuracy" not in PROBA_METRICS


# ---------------------------------------------------------------------------
# T5-02: needs_proba
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", list(LABEL_METRICS))
def test_needs_proba_false_for_label_metrics(name: str) -> None:
    assert needs_proba(name) is False


@pytest.mark.parametrize("name", list(PROBA_METRICS))
def test_needs_proba_true_for_proba_metrics(name: str) -> None:
    assert needs_proba(name) is True


def test_needs_proba_unknown_raises() -> None:
    with pytest.raises(UndefinedMetric):
        needs_proba("nonexistent_metric")


# ---------------------------------------------------------------------------
# T5-03: metric_fn unknown name
# ---------------------------------------------------------------------------


def test_metric_fn_unknown_raises() -> None:
    with pytest.raises(UndefinedMetric):
        metric_fn("bad_metric")


# ---------------------------------------------------------------------------
# T5-04: Hand-computed small example
# ---------------------------------------------------------------------------


def _hand_example() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    # 4 samples: TP, FP, FN, TN
    y_true = np.array([1, 0, 1, 0])
    y_pred = np.array([1, 1, 0, 0])
    y_proba = np.array([0.9, 0.8, 0.3, 0.1])
    return y_true, y_pred, y_proba


def test_accuracy_hand() -> None:
    yt, yp, ypr = _hand_example()
    fn = metric_fn("accuracy")
    assert fn(yt, yp, ypr) == pytest.approx(0.5)  # 2/4 correct


def test_f1_hand() -> None:
    yt, yp, ypr = _hand_example()
    # TP=1, FP=1, FN=1 → precision=0.5, recall=0.5, f1=0.5
    fn = metric_fn("f1")
    assert fn(yt, yp, ypr) == pytest.approx(0.5)


def test_precision_hand() -> None:
    yt, yp, ypr = _hand_example()
    fn = metric_fn("precision")
    assert fn(yt, yp, ypr) == pytest.approx(0.5)


def test_recall_hand() -> None:
    yt, yp, ypr = _hand_example()
    fn = metric_fn("recall")
    assert fn(yt, yp, ypr) == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# T5-05: sklearn parity on random data
# ---------------------------------------------------------------------------


def _random_binary(n: int = 200, seed: int = 42) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    y_true = rng.integers(0, 2, size=n)
    y_pred = rng.integers(0, 2, size=n)
    y_proba = rng.random(size=n)
    return y_true, y_pred, y_proba


def test_sklearn_parity_accuracy() -> None:
    yt, yp, ypr = _random_binary()
    result = metric_fn("accuracy")(yt, yp, ypr)
    expected = accuracy_score(yt, yp)
    assert abs(result - expected) < 1e-12


def test_sklearn_parity_balanced_accuracy() -> None:
    yt, yp, ypr = _random_binary()
    result = metric_fn("balanced_accuracy")(yt, yp, ypr)
    expected = balanced_accuracy_score(yt, yp)
    assert abs(result - expected) < 1e-12


def test_sklearn_parity_macro_f1() -> None:
    yt, yp, ypr = _random_binary()
    result = metric_fn("macro_f1")(yt, yp, ypr)
    expected = f1_score(yt, yp, average="macro", zero_division=0)
    assert abs(result - expected) < 1e-12


def test_sklearn_parity_f1() -> None:
    yt, yp, ypr = _random_binary()
    result = metric_fn("f1")(yt, yp, ypr)
    expected = f1_score(yt, yp, average="binary", pos_label=1, zero_division=0)
    assert abs(result - expected) < 1e-12


def test_sklearn_parity_roc_auc() -> None:
    yt, yp, ypr = _random_binary()
    result = metric_fn("roc_auc")(yt, yp, ypr)
    expected = roc_auc_score(yt, ypr)
    assert abs(result - expected) < 1e-12


def test_sklearn_parity_pr_auc() -> None:
    yt, yp, ypr = _random_binary()
    result = metric_fn("pr_auc")(yt, yp, ypr)
    expected = average_precision_score(yt, ypr)
    assert abs(result - expected) < 1e-12


# ---------------------------------------------------------------------------
# T5-06: Single-class target — accuracy OK, others raise
# ---------------------------------------------------------------------------


def test_accuracy_single_class_ok() -> None:
    yt = np.array([1, 1, 1, 1])
    yp = np.array([1, 1, 1, 1])
    ypr = np.array([0.9, 0.8, 0.7, 0.6])
    fn = metric_fn("accuracy")
    assert fn(yt, yp, ypr) == pytest.approx(1.0)


@pytest.mark.parametrize(
    "name",
    ["balanced_accuracy", "macro_f1", "f1", "precision", "recall", "roc_auc", "pr_auc"],
)
def test_single_class_raises_undefined(name: str) -> None:
    yt = np.array([0, 0, 0, 0])
    yp = np.array([0, 0, 0, 0])
    ypr = np.array([0.1, 0.2, 0.3, 0.1])
    fn = metric_fn(name)
    with pytest.raises(UndefinedMetric):
        fn(yt, yp, ypr)


# ---------------------------------------------------------------------------
# T5-07: compute_metrics validation
# ---------------------------------------------------------------------------


def test_compute_metrics_empty_raises() -> None:
    with pytest.raises(ValueError, match="Empty"):
        compute_metrics(np.array([]), np.array([]), np.array([]))


def test_compute_metrics_length_mismatch_raises() -> None:
    with pytest.raises(ValueError):
        compute_metrics(np.array([0, 1, 0]), np.array([0, 1]), np.array([0.1, 0.9, 0.2]))


def test_compute_metrics_invalid_label_raises() -> None:
    with pytest.raises(ValueError, match="0 and 1"):
        compute_metrics(np.array([0, 2, 1]), np.array([0, 1, 1]), np.array([0.1, 0.5, 0.9]))


def test_compute_metrics_proba_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="\\[0, 1\\]"):
        compute_metrics(
            np.array([0, 1, 0, 1]), np.array([0, 1, 0, 1]), np.array([0.1, 1.5, 0.3, 0.9])
        )


def test_compute_metrics_non_finite_proba_raises() -> None:
    with pytest.raises(ValueError, match="non-finite"):
        compute_metrics(np.array([0, 1]), np.array([0, 1]), np.array([float("nan"), 0.5]))


def test_compute_metrics_returns_all_names() -> None:
    yt, yp, ypr = _random_binary(n=100)
    result = compute_metrics(yt, yp, ypr)
    assert set(result.keys()) == set(METRIC_NAMES)
