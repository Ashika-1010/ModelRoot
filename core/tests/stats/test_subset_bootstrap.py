"""Tests for paired_subset_bootstrap (M15)."""

import numpy as np
import pytest

from modelroot.stats.bootstrap import BootstrapError, BootstrapResult
from modelroot.stats.subset_bootstrap import paired_subset_bootstrap


def test_paired_subset_bootstrap_basic() -> None:
    """Validate delta = full - subset, metric_a=subset, metric_b=full."""
    # 20 samples, subset is first 10
    y_true = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    pred = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
    # Subset: accuracy 1.0 (10/10)
    # Full: accuracy 1.0 (20/20)
    mask = np.zeros(20, dtype=bool)
    mask[:10] = True

    res = paired_subset_bootstrap(y_true, pred, mask, metric="accuracy", n_boot=200, seed=42)
    assert isinstance(res, BootstrapResult)
    assert np.isclose(res.metric_a, 1.0)
    assert np.isclose(res.metric_b, 1.0)
    assert np.isclose(res.delta, 0.0)
    assert res.ci_low <= res.delta <= res.ci_high
    assert res.se >= 0.0


def test_paired_subset_bootstrap_inflation() -> None:
    """Validate positive delta when full set score is inflated compared to subset."""
    # Subset: 50% accuracy; Full: 75% accuracy
    y_true = np.array([1, 0, 1, 0, 1, 1, 1, 1])
    pred = np.array([1, 1, 0, 0, 1, 1, 1, 1])
    mask = np.array([True, True, True, True, False, False, False, False])

    res = paired_subset_bootstrap(y_true, pred, mask, metric="accuracy", n_boot=200, seed=42)
    # subset accuracy = 2/4 = 0.5
    # full accuracy = 6/8 = 0.75
    # delta = full - subset = 0.25
    assert np.isclose(res.metric_a, 0.5)
    assert np.isclose(res.metric_b, 0.75)
    assert np.isclose(res.delta, 0.25)


def test_paired_subset_bootstrap_validation() -> None:
    """Validate input checks: lengths, binary labels, mask count."""
    y = np.array([1, 0, 1, 0])
    p = np.array([1, 0, 1])
    m = np.array([True, True, True, False])

    # Length mismatch
    with pytest.raises(ValueError, match="same length"):
        paired_subset_bootstrap(y, p, m, metric="accuracy")

    # Non-binary labels
    with pytest.raises(ValueError, match="binary labels"):
        paired_subset_bootstrap(np.array([1, 2, 0, 1]), y, m, metric="accuracy")

    # < 2 subset rows
    m_few = np.array([True, False, False, False])
    with pytest.raises(ValueError, match="at least two True rows"):
        paired_subset_bootstrap(y, y, m_few, metric="accuracy")


def test_paired_subset_bootstrap_invalid_resample_limit() -> None:
    """Validate BootstrapError when subset lacks class diversity in resamples."""
    # Subset has only one positive class member among 100 samples
    # -> resamples will frequently lack 0 or 1
    y = np.array([1] + [0] * 50)
    p = np.array([1] + [0] * 50)
    mask = np.zeros(51, dtype=bool)
    mask[0] = True  # the single positive class
    mask[1] = True  # one negative class

    # AUC metric on subset with 2 items will fail almost every resample
    with pytest.raises(BootstrapError, match="invalid"):
        paired_subset_bootstrap(y, p, mask, metric="roc_auc", n_boot=200, y_proba=p.astype(float))
