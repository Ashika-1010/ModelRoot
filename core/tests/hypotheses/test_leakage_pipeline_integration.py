"""Integration tests exercising the actual diagnose() pipeline for M15/M16 hypotheses.

Tests verify end-to-end execution through diagnose():
1. Identifier leakage S2 and S3 -> supported
2. Plain target-proxy S2 and S3 -> weakly_supported (never supported)
3. Flagged target-proxy S3f -> supported (importance share >= 0.10)
4. Transfer target-proxy S3t -> supported (all 3 transfer conditions pass)
5. Legitimate strong-feature hard negative -> weakly_supported (never supported)

Also verifies:
- No hypotheses are silently dropped due to runtime exceptions.
- Mandatory H3c verbatim caveat is present.
- Real dictionary metadata is supported without AttributeError.
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np
import pandas as pd
import pytest

from modelroot.engine.pipeline import diagnose
from modelroot.hypotheses.leakage import (
    LeakageIdentifierHypothesis,
    LeakageTargetProxyHypothesis,
)
from modelroot.schemas.common import Status
from modelroot.schemas.config import AnalysisConfig

H3C_MANDATORY_CAVEAT = (
    "ModelRoot cannot distinguish proxy leakage from a legitimately strong feature "
    "without knowing whether the feature is available at prediction time; "
    "this finding requires domain review."
)


def _assert_no_hypothesis_failures(result: Any) -> None:
    """Ensure no hypothesis crashed silently during diagnose()."""
    failure_warnings = [w for w in result.warnings if "Hypothesis " in w and "failed" in w]
    assert not failure_warnings, f"Hypothesis failed unexpectedly in pipeline: {failure_warnings}"


# ---------------------------------------------------------------------------
# 1. Identifier leakage S2 and S3 -> supported
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("severity,sort_frac", [("S2", 0.90), ("S3", 1.00)])
def test_pipeline_identifier_leakage(
    tmp_path: pathlib.Path, severity: str, sort_frac: float
) -> None:
    """Identifier leakage S2 and S3 must execute without AttributeError and produce supported."""
    rng = np.random.default_rng(42)
    n = 1000
    y = np.array([0] * (n // 2) + [1] * (n // 2))

    # Add noise features
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    # Sort rows by target according to sort_frac
    df = pd.DataFrame({"x1": x1, "x2": x2, "target": y})
    df_sorted = df.sort_values(by="target").reset_index(drop=True)

    if sort_frac < 1.0:
        # Partially shuffle (1 - sort_frac) of the rows
        n_shuffle = int(n * (1.0 - sort_frac))
        shuf_idx = rng.choice(n, size=n_shuffle, replace=False)
        permuted_idx = rng.permutation(shuf_idx)
        df_sorted.iloc[shuf_idx] = df_sorted.iloc[permuted_idx].values

    # Add integer row_id
    df_sorted["row_id"] = np.arange(n, dtype=int)

    csv_path = tmp_path / f"ident_{severity}.csv"
    df_sorted.to_csv(csv_path, index=False)

    config = AnalysisConfig(
        data_path=str(csv_path),
        target="target",
        positive_class="1",
        split={"mode": "single_file", "test_size": 0.3},
        model_spec={"algorithm": "decision_tree", "params": {"max_depth": 5}},
        seed=42,
        budget_s=120,
    )

    result = diagnose(config, hypotheses=[LeakageIdentifierHypothesis])
    _assert_no_hypothesis_failures(result)

    assert result.status == "completed"
    ident_res = [
        h for h in result.hypotheses if h.subtype == "identifier" or h.id == "leakage_identifier"
    ]
    assert len(ident_res) == 1, f"Expected 1 identifier hypothesis result, got {len(ident_res)}"
    assert ident_res[0].status == Status.supported, (
        f"Expected identifier {severity} to be supported, got {ident_res[0].status.value}"
    )


# ---------------------------------------------------------------------------
# 2. Plain target-proxy S2 and S3 -> weakly_supported (never supported)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("severity,noise_sigma", [("S2", 0.3), ("S3", 0.0)])
def test_pipeline_plain_target_proxy(
    tmp_path: pathlib.Path, severity: str, noise_sigma: float
) -> None:
    """Plain unflagged target-proxy S2/S3 must produce weakly_supported, NEVER supported."""
    rng = np.random.default_rng(101)
    n = 1000
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rng.shuffle(y)

    proxy = y.astype(float) + (rng.normal(0, noise_sigma, n) if noise_sigma > 0 else 0.0)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "proxy": proxy, "target": y})
    csv_path = tmp_path / f"plain_proxy_{severity}.csv"
    df.to_csv(csv_path, index=False)

    config = AnalysisConfig(
        data_path=str(csv_path),
        target="target",
        positive_class="1",
        split={"mode": "single_file", "test_size": 0.3},
        model_spec={"algorithm": "decision_tree", "params": {"max_depth": 5}},
        seed=42,
        budget_s=120,
        metadata={},  # unflagged dictionary
    )

    result = diagnose(config, hypotheses=[LeakageTargetProxyHypothesis])
    _assert_no_hypothesis_failures(result)

    proxy_res = [
        h
        for h in result.hypotheses
        if h.subtype == "target_proxy" or h.id == "leakage_target_proxy"
    ]
    assert len(proxy_res) == 1
    assert proxy_res[0].status != Status.supported, (
        f"Plain unflagged proxy must NEVER be supported, got {proxy_res[0].status.value}"
    )
    if severity == "S3":
        assert proxy_res[0].status == Status.weakly_supported
    else:
        assert proxy_res[0].status in (Status.weakly_supported, Status.inconclusive)
    assert any(H3C_MANDATORY_CAVEAT in c for c in proxy_res[0].caveats)


# ---------------------------------------------------------------------------
# 3. Flagged target-proxy S3f -> supported (importance share >= 0.10)
# ---------------------------------------------------------------------------


def test_pipeline_flagged_target_proxy_s3f(tmp_path: pathlib.Path) -> None:
    """Flagged S3f proxy produces supported when importance share meets the 0.10 threshold."""
    rng = np.random.default_rng(202)
    n = 1000
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rng.shuffle(y)

    proxy = y.astype(float)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "proxy": proxy, "target": y})
    csv_path = tmp_path / "flagged_proxy_s3f.csv"
    df.to_csv(csv_path, index=False)

    # metadata as dictionary with unavailable_at_inference
    config = AnalysisConfig(
        data_path=str(csv_path),
        target="target",
        positive_class="1",
        split={"mode": "single_file", "test_size": 0.3},
        model_spec={"algorithm": "decision_tree", "params": {"max_depth": 5}},
        seed=42,
        budget_s=120,
        metadata={"unavailable_at_inference": ["proxy"]},
    )

    result = diagnose(config, hypotheses=[LeakageTargetProxyHypothesis])
    _assert_no_hypothesis_failures(result)

    proxy_res = [
        h
        for h in result.hypotheses
        if h.subtype == "target_proxy" or h.id == "leakage_target_proxy"
    ]
    assert len(proxy_res) == 1
    assert proxy_res[0].status == Status.supported, (
        f"Flagged S3 proxy must be supported, got {proxy_res[0].status.value}"
    )
    assert any(H3C_MANDATORY_CAVEAT in c for c in proxy_res[0].caveats)


# ---------------------------------------------------------------------------
# 4. Transfer target-proxy S3t -> supported
# ---------------------------------------------------------------------------


def test_pipeline_transfer_target_proxy_s3t(tmp_path: pathlib.Path) -> None:
    """Transfer S3t produces supported when all three transfer conditions pass."""
    rng = np.random.default_rng(303)
    n_base = 1000
    y_base = np.array([0] * (n_base // 2) + [1] * (n_base // 2))
    rng.shuffle(y_base)

    proxy_base = y_base.astype(float)
    x1_base = rng.standard_normal(n_base)
    x2_base = rng.standard_normal(n_base)
    df_base = pd.DataFrame({"x1": x1_base, "x2": x2_base, "proxy": proxy_base, "target": y_base})

    base_path = tmp_path / "transfer_base.csv"
    df_base.to_csv(base_path, index=False)

    # Current split has broken proxy (pure noise uncorrelated with target)
    n_curr = 400
    y_curr = np.array([0] * (n_curr // 2) + [1] * (n_curr // 2))
    rng.shuffle(y_curr)
    proxy_curr = rng.standard_normal(n_curr)
    x1_curr = rng.standard_normal(n_curr)
    x2_curr = rng.standard_normal(n_curr)
    df_curr = pd.DataFrame({"x1": x1_curr, "x2": x2_curr, "proxy": proxy_curr, "target": y_curr})

    curr_path = tmp_path / "transfer_current.csv"
    df_curr.to_csv(curr_path, index=False)

    config = AnalysisConfig(
        data_path=str(base_path),
        target="target",
        positive_class="1",
        split={"mode": "single_file", "test_size": 0.3},
        model_spec={"algorithm": "decision_tree", "params": {"max_depth": 5}},
        seed=42,
        budget_s=120,
        current_path=str(curr_path),
        metadata={},  # unflagged, relying entirely on transfer diagnostic
    )

    result = diagnose(config, hypotheses=[LeakageTargetProxyHypothesis])
    _assert_no_hypothesis_failures(result)

    proxy_res = [
        h
        for h in result.hypotheses
        if h.subtype == "target_proxy" or h.id == "leakage_target_proxy"
    ]
    assert len(proxy_res) == 1
    assert proxy_res[0].status == Status.supported, (
        f"Transfer S3t proxy must be supported, got {proxy_res[0].status.value}"
    )
    assert any(H3C_MANDATORY_CAVEAT in c for c in proxy_res[0].caveats)


# ---------------------------------------------------------------------------
# 5. Legitimate strong-feature hard negative -> weakly_supported (never supported)
# ---------------------------------------------------------------------------


def test_pipeline_hard_negative_strong_feature(tmp_path: pathlib.Path) -> None:
    """Legitimate strong feature is never supported solely because of predictive strength."""
    rng = np.random.default_rng(404)
    n = 1000
    y = np.array([0] * (n // 2) + [1] * (n // 2))
    rng.shuffle(y)

    # Legitimate predictive feature (e.g. noise std = 0.35)
    strong_feat = y.astype(float) + rng.normal(0, 0.35, n)
    x1 = rng.standard_normal(n)
    x2 = rng.standard_normal(n)

    df = pd.DataFrame({"x1": x1, "x2": x2, "strong_feat": strong_feat, "target": y})
    csv_path = tmp_path / "hard_negative_strong.csv"
    df.to_csv(csv_path, index=False)

    config = AnalysisConfig(
        data_path=str(csv_path),
        target="target",
        positive_class="1",
        split={"mode": "single_file", "test_size": 0.3},
        model_spec={"algorithm": "decision_tree", "params": {"max_depth": 5}},
        seed=42,
        budget_s=120,
        metadata={},  # not flagged
    )

    result = diagnose(config, hypotheses=[LeakageTargetProxyHypothesis])
    _assert_no_hypothesis_failures(result)

    proxy_res = [
        h
        for h in result.hypotheses
        if h.subtype == "target_proxy" or h.id == "leakage_target_proxy"
    ]
    assert len(proxy_res) == 1
    assert proxy_res[0].status != Status.supported, (
        f"Legitimate strong feature must NEVER be supported, got {proxy_res[0].status.value}"
    )
    assert proxy_res[0].status in (Status.weakly_supported, Status.inconclusive)
    assert any(H3C_MANDATORY_CAVEAT in c for c in proxy_res[0].caveats)
