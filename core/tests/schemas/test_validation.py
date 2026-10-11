"""Validation-rule tests: boundary checks and rejection of invalid inputs.

Covers every constraint specified in §3.2 and the M1 prompt:
- ci_low > ci_high rejected (MetricResult, Evidence, EffectResult, NBResult,
  ExperimentResult, HypothesisResult.effect)
- evidence_strength outside [0, 1] rejected
- Empty IDs rejected
- seed < 0 rejected
- budget_s <= 0 rejected
- Non-scalar params rejected
- Invalid hypothesis/subtype combinations rejected
- SplitConfig cross-field constraints
- AnalysisResult.status must be "completed" | "partial" | "failed"
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelroot.schemas import (
    AnalysisConfig,
    AnalysisResult,
    EffectResult,
    Evidence,
    ExperimentResult,
    ExperimentSpec,
    HypothesisResult,
    MetricResult,
    ModelSpec,
    NBResult,
    SplitConfig,
)

# ---------------------------------------------------------------------------
# Helpers duplicated here to keep tests self-contained
# ---------------------------------------------------------------------------


def _base_metric() -> dict:  # type: ignore[type-arg]
    return {"name": "macro_f1", "value": 0.75, "split": "holdout", "n": 200}


def _base_evidence() -> dict:  # type: ignore[type-arg]
    return {
        "id": "ev_abcd1234",
        "kind": "diagnostic",
        "name": "gap",
        "value": 0.19,
        "method": "metric_difference",
        "hint": "Train-holdout gap.",
    }


def _base_exp_spec() -> dict:  # type: ignore[type-arg]
    return {
        "id": "exp-001",
        "hypothesis_id": "hyp-001",
        "name": "Regularize",
        "intervention": "regularize",
        "metric": "macro_f1",
        "seed": 0,
        "budget_s": 60.0,
    }


def _base_exp_result() -> dict:  # type: ignore[type-arg]
    return {
        "spec_id": "exp-001",
        "status": "completed",
        "baseline_metric": 0.72,
        "intervention_metric": 0.78,
        "delta": 0.06,
        "ci_low": 0.01,
        "ci_high": 0.11,
        "n_eval": 400,
        "n_valid_boot": 1000,
        "runtime_s": 5.2,
    }


def _base_hyp() -> dict:  # type: ignore[type-arg]
    return {
        "id": "hyp-001",
        "type": "overfitting",
        "status": "supported",
        "evidence_strength": 0.85,
        "trigger": {"fired": True, "rule_id": "OF-T1"},
        "decision": {
            "rule_id": "OF-D1",
            "rule_text": "rule",
            "rules_profile": "full",
        },
    }


def _base_effect() -> dict:  # type: ignore[type-arg]
    return {
        "metric": "macro_f1",
        "baseline": 0.72,
        "intervention": 0.78,
        "delta": 0.06,
        "ci_low": 0.01,
        "ci_high": 0.11,
        "n": 400,
    }


def _base_analysis_config() -> dict:  # type: ignore[type-arg]
    return {
        "data_path": "data/x.csv",
        "target": "label",
        "positive_class": "1",
        "model_spec": {"algorithm": "logistic_regression", "params": {}},
    }


def _base_analysis_result() -> dict:  # type: ignore[type-arg]
    return {
        "analysis_id": "run-001",
        "created_at": "2026-10-10T12:00:00Z",
        "status": "completed",
        "config": _base_analysis_config(),
        "dataset_fingerprint": "fp-ds",
        "split_fingerprint": "fp-sp",
        "rule_set_version": "1.0",
        "assessment": {
            "reference": {"algorithm": "random_forest"},
            "headroom": 0.09,
            "train_holdout_gap": 0.19,
            "underperforming": True,
            "headline": "Underperforming.",
        },
        "reproducibility_hash": "hash-xyz",
    }


# ---------------------------------------------------------------------------
# CI ordering: ci_low > ci_high must be rejected
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "model_cls,base_fn,ci_fields",
    [
        (MetricResult, _base_metric, ("ci_low", "ci_high")),
        (Evidence, _base_evidence, ("ci_low", "ci_high")),
        (EffectResult, _base_effect, ("ci_low", "ci_high")),
    ],
)
def test_ci_ordering_rejected(model_cls, base_fn, ci_fields):  # type: ignore[no-untyped-def]
    lo, hi = ci_fields
    data = {**base_fn(), lo: 0.9, hi: 0.1}
    with pytest.raises(ValidationError, match="ci_low"):
        model_cls.model_validate(data)


def test_nb_result_ci_ordering_rejected() -> None:
    with pytest.raises(ValidationError, match="ci_low"):
        NBResult.model_validate(
            {
                "mean": 0.03,
                "se": 0.01,
                "t": 3.0,
                "df": 9.0,
                "p": 0.014,
                "ci_low": 0.9,
                "ci_high": 0.1,
            }
        )


def test_experiment_result_ci_ordering_rejected() -> None:
    with pytest.raises(ValidationError, match="ci_low"):
        ExperimentResult.model_validate({**_base_exp_result(), "ci_low": 0.9, "ci_high": 0.1})


def test_ci_equal_is_accepted() -> None:
    """ci_low == ci_high is a degenerate but valid CI (zero-width)."""
    obj = MetricResult.model_validate({**_base_metric(), "ci_low": 0.5, "ci_high": 0.5})
    assert obj.ci_low == obj.ci_high == 0.5


def test_ci_only_low_is_accepted() -> None:
    """Providing only ci_low (without ci_high) must not raise."""
    obj = MetricResult.model_validate({**_base_metric(), "ci_low": 0.5})
    assert obj.ci_low == 0.5
    assert obj.ci_high is None


def test_ci_only_high_is_accepted() -> None:
    """Providing only ci_high (without ci_low) must not raise."""
    obj = MetricResult.model_validate({**_base_metric(), "ci_high": 0.8})
    assert obj.ci_high == 0.8
    assert obj.ci_low is None


# ---------------------------------------------------------------------------
# evidence_strength bounds
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_strength", [-0.001, -1.0, 1.001, 2.0, 100.0])
def test_evidence_strength_out_of_range_rejected(bad_strength: float) -> None:
    with pytest.raises(ValidationError, match="evidence_strength"):
        HypothesisResult.model_validate({**_base_hyp(), "evidence_strength": bad_strength})


@pytest.mark.parametrize("good_strength", [0.0, 0.5, 1.0])
def test_evidence_strength_boundaries_accepted(good_strength: float) -> None:
    obj = HypothesisResult.model_validate({**_base_hyp(), "evidence_strength": good_strength})
    assert obj.evidence_strength == good_strength


# ---------------------------------------------------------------------------
# Non-empty IDs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("empty_id", ["", "   ", "\t"])
def test_evidence_empty_id_rejected(empty_id: str) -> None:
    with pytest.raises(ValidationError):
        Evidence.model_validate({**_base_evidence(), "id": empty_id})


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_experiment_spec_empty_id_rejected(empty_id: str) -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({**_base_exp_spec(), "id": empty_id})


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_experiment_spec_empty_hypothesis_id_rejected(empty_id: str) -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({**_base_exp_spec(), "hypothesis_id": empty_id})


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_hypothesis_result_empty_id_rejected(empty_id: str) -> None:
    with pytest.raises(ValidationError):
        HypothesisResult.model_validate({**_base_hyp(), "id": empty_id})


@pytest.mark.parametrize("empty_id", ["", "   "])
def test_analysis_result_empty_analysis_id_rejected(empty_id: str) -> None:
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**_base_analysis_result(), "analysis_id": empty_id})


# ---------------------------------------------------------------------------
# seed >= 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_seed", [-1, -100])
def test_analysis_config_negative_seed_rejected(bad_seed: int) -> None:
    with pytest.raises(ValidationError, match="seed"):
        AnalysisConfig.model_validate({**_base_analysis_config(), "seed": bad_seed})


@pytest.mark.parametrize("bad_seed", [-1, -100])
def test_experiment_spec_negative_seed_rejected(bad_seed: int) -> None:
    with pytest.raises(ValidationError, match="seed"):
        ExperimentSpec.model_validate({**_base_exp_spec(), "seed": bad_seed})


def test_seed_zero_is_valid() -> None:
    obj = AnalysisConfig.model_validate({**_base_analysis_config(), "seed": 0})
    assert obj.seed == 0


# ---------------------------------------------------------------------------
# budget_s > 0
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_budget", [0, -1, -100.0])
def test_analysis_config_non_positive_budget_rejected(bad_budget: float) -> None:
    with pytest.raises(ValidationError, match="budget_s"):
        AnalysisConfig.model_validate({**_base_analysis_config(), "budget_s": bad_budget})


@pytest.mark.parametrize("bad_budget", [0, -1.0])
def test_experiment_spec_non_positive_budget_rejected(bad_budget: float) -> None:
    with pytest.raises(ValidationError, match="budget_s"):
        ExperimentSpec.model_validate({**_base_exp_spec(), "budget_s": bad_budget})


# ---------------------------------------------------------------------------
# Non-scalar params
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_value",
    [
        [1, 2, 3],  # list
        {"nested": "dict"},  # dict
        object(),  # arbitrary object
    ],
)
def test_model_spec_non_scalar_param_rejected(bad_value: object) -> None:
    with pytest.raises((ValidationError, Exception)):
        ModelSpec.model_validate({"algorithm": "logistic_regression", "params": {"x": bad_value}})


@pytest.mark.parametrize(
    "bad_value",
    [
        [1, 2],
        {"k": "v"},
    ],
)
def test_experiment_spec_non_scalar_param_rejected(bad_value: object) -> None:
    with pytest.raises((ValidationError, Exception)):
        ExperimentSpec.model_validate({**_base_exp_spec(), "params": {"p": bad_value}})


def test_scalar_params_accepted() -> None:
    """int, float, str, bool, and None are all valid param values."""
    obj = ModelSpec.model_validate(
        {
            "algorithm": "logistic_regression",
            "params": {
                "C": 1.0,
                "max_iter": 200,
                "penalty": "l2",
                "fit_intercept": True,
                "tol": None,
            },
        }
    )
    assert obj.params["C"] == 1.0
    assert obj.params["tol"] is None


# ---------------------------------------------------------------------------
# Hypothesis / subtype combinations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "hyp_type,bad_subtype",
    [
        # overfitting has no subtypes
        ("overfitting", "distortion"),
        ("overfitting", "duplicate"),
        # imbalance only allows distortion / harm
        ("imbalance", "duplicate"),
        ("imbalance", "identifier"),
        ("imbalance", "representation_related"),
        # leakage only allows duplicate / identifier / target_proxy
        ("leakage", "distortion"),
        ("leakage", "harm"),
        ("leakage", "representation_related"),
        # subgroup allows representation_related / intrinsic_or_unresolved / None
        ("subgroup", "distortion"),
        ("subgroup", "duplicate"),
        # shift has no subtypes
        ("shift", "harm"),
        ("shift", "duplicate"),
    ],
)
def test_invalid_subtype_combination_rejected(hyp_type: str, bad_subtype: str) -> None:
    with pytest.raises(ValidationError, match="subtype"):
        HypothesisResult.model_validate({**_base_hyp(), "type": hyp_type, "subtype": bad_subtype})


@pytest.mark.parametrize(
    "hyp_type,good_subtype",
    [
        ("overfitting", None),
        ("imbalance", "distortion"),
        ("imbalance", "harm"),
        ("leakage", "duplicate"),
        ("leakage", "identifier"),
        ("leakage", "target_proxy"),
        ("subgroup", "representation_related"),
        ("subgroup", "intrinsic_or_unresolved"),
        ("subgroup", None),
        ("shift", None),
    ],
)
def test_valid_subtype_combination_accepted(hyp_type: str, good_subtype: object) -> None:
    obj = HypothesisResult.model_validate(
        {**_base_hyp(), "type": hyp_type, "subtype": good_subtype}
    )
    assert obj.subtype == good_subtype


# ---------------------------------------------------------------------------
# SplitConfig cross-field constraints
# ---------------------------------------------------------------------------


def test_split_column_mode_requires_split_column() -> None:
    with pytest.raises(ValidationError, match="split_column"):
        SplitConfig.model_validate({"mode": "split_column"})


def test_split_column_mode_accepts_column_name() -> None:
    obj = SplitConfig.model_validate({"mode": "split_column", "split_column": "fold"})
    assert obj.split_column == "fold"


def test_test_file_mode_requires_test_path() -> None:
    with pytest.raises(ValidationError, match="test_path"):
        SplitConfig.model_validate({"mode": "test_file"})


def test_test_file_mode_accepts_path() -> None:
    obj = SplitConfig.model_validate({"mode": "test_file", "test_path": "data/test.csv"})
    assert obj.test_path is not None


def test_test_size_out_of_range_rejected() -> None:
    with pytest.raises(ValidationError, match="test_size"):
        SplitConfig.model_validate({"mode": "single_file", "test_size": 0.0})
    with pytest.raises(ValidationError, match="test_size"):
        SplitConfig.model_validate({"mode": "single_file", "test_size": 1.0})
    with pytest.raises(ValidationError, match="test_size"):
        SplitConfig.model_validate({"mode": "single_file", "test_size": 1.5})


# ---------------------------------------------------------------------------
# AnalysisResult.status
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("bad_status", ["running", "pending", "ok", "done", ""])
def test_analysis_result_invalid_status_rejected(bad_status: str) -> None:
    with pytest.raises(ValidationError, match="status"):
        AnalysisResult.model_validate({**_base_analysis_result(), "status": bad_status})


@pytest.mark.parametrize("good_status", ["completed", "partial", "failed"])
def test_analysis_result_valid_statuses_accepted(good_status: str) -> None:
    obj = AnalysisResult.model_validate({**_base_analysis_result(), "status": good_status})
    assert obj.status == good_status
