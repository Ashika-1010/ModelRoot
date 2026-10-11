"""Round-trip and unknown-field tests for every public ModelRoot contract model.

Tests verify that:
- Every model can be constructed from a minimal valid dict.
- ``model_dump()`` round-trips back through ``model_validate()`` without loss.
- Every model rejects unknown fields with a ``ValidationError``.
- JSON serialisation / deserialisation is lossless (via ``model_dump_json`` /
  ``model_validate_json``).
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from modelroot.schemas import (
    SCHEMA_VERSION,
    AnalysisConfig,
    AnalysisResult,
    AssessmentResult,
    ColumnProfile,
    DatasetProfile,
    Decision,
    EffectResult,
    Evidence,
    ExperimentResult,
    ExperimentSpec,
    ExplanationOutput,
    HypothesisResult,
    Metadata,
    MetricResult,
    ModelSpec,
    NBResult,
    PrimaryMetric,
    ReferenceResult,
    RulesProfile,
    SplitConfig,
    TriggerResult,
)
from modelroot.schemas.config import AblationConfig

# ---------------------------------------------------------------------------
# Minimal valid construction helpers
# ---------------------------------------------------------------------------


def _model_spec() -> dict:  # type: ignore[type-arg]
    return {"algorithm": "logistic_regression", "params": {"C": 1.0}}


def _split_config() -> dict:  # type: ignore[type-arg]
    return {"mode": "single_file"}


def _analysis_config() -> dict:  # type: ignore[type-arg]
    return {
        "data_path": "data/x.csv",
        "target": "label",
        "positive_class": "1",
        "model_spec": _model_spec(),
    }


def _metric_result(split: str = "holdout") -> dict:  # type: ignore[type-arg]
    return {"name": "macro_f1", "value": 0.75, "split": split, "n": 200}


def _evidence() -> dict:  # type: ignore[type-arg]
    return {
        "id": "ev_abcd1234",
        "kind": "diagnostic",
        "name": "gap",
        "value": 0.19,
        "method": "metric_difference",
        "hint": "Train-holdout gap.",
    }


def _experiment_spec() -> dict:  # type: ignore[type-arg]
    return {
        "id": "exp-001",
        "hypothesis_id": "hyp-001",
        "name": "Regularize",
        "intervention": "regularize",
        "metric": "macro_f1",
        "seed": 0,
        "budget_s": 60.0,
    }


def _nb_result() -> dict:  # type: ignore[type-arg]
    return {
        "mean": 0.03,
        "se": 0.01,
        "t": 3.0,
        "df": 9.0,
        "p": 0.014,
        "ci_low": 0.01,
        "ci_high": 0.05,
    }


def _experiment_result() -> dict:  # type: ignore[type-arg]
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


def _trigger() -> dict:  # type: ignore[type-arg]
    return {"fired": True, "rule_id": "OF-T1"}


def _decision() -> dict:  # type: ignore[type-arg]
    return {
        "rule_id": "OF-D1",
        "rule_text": "supported if delta>=delta_min",
        "rules_profile": "full",
    }


def _effect() -> dict:  # type: ignore[type-arg]
    return {
        "metric": "macro_f1",
        "baseline": 0.72,
        "intervention": 0.78,
        "delta": 0.06,
        "ci_low": 0.01,
        "ci_high": 0.11,
        "n": 400,
    }


def _hypothesis_result() -> dict:  # type: ignore[type-arg]
    return {
        "id": "hyp-001",
        "type": "overfitting",
        "status": "supported",
        "evidence_strength": 0.85,
        "trigger": _trigger(),
        "decision": _decision(),
    }


def _reference_result() -> dict:  # type: ignore[type-arg]
    return {"algorithm": "random_forest", "metrics": [_metric_result()]}


def _assessment() -> dict:  # type: ignore[type-arg]
    return {
        "reference": _reference_result(),
        "headroom": 0.09,
        "train_holdout_gap": 0.19,
        "underperforming": True,
        "headline": "Model is underperforming.",
    }


def _analysis_result() -> dict:  # type: ignore[type-arg]
    return {
        "analysis_id": "run-001",
        "created_at": "2026-10-10T12:00:00Z",
        "status": "completed",
        "config": _analysis_config(),
        "dataset_fingerprint": "fp-dataset",
        "split_fingerprint": "fp-split",
        "rule_set_version": "1.0",
        "assessment": _assessment(),
        "reproducibility_hash": "hash-abc",
    }


# ---------------------------------------------------------------------------
# ModelSpec
# ---------------------------------------------------------------------------


def test_model_spec_round_trip() -> None:
    obj = ModelSpec.model_validate(_model_spec())
    assert ModelSpec.model_validate(obj.model_dump()) == obj


def test_model_spec_json_round_trip() -> None:
    obj = ModelSpec.model_validate(_model_spec())
    assert ModelSpec.model_validate_json(obj.model_dump_json()) == obj


def test_model_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ModelSpec.model_validate({**_model_spec(), "extra_field": "x"})


# ---------------------------------------------------------------------------
# SplitConfig
# ---------------------------------------------------------------------------


def test_split_config_round_trip() -> None:
    obj = SplitConfig.model_validate(_split_config())
    assert SplitConfig.model_validate(obj.model_dump()) == obj


def test_split_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        SplitConfig.model_validate({**_split_config(), "foo": 1})


# ---------------------------------------------------------------------------
# Metadata
# ---------------------------------------------------------------------------


def test_metadata_round_trip() -> None:
    obj = Metadata.model_validate({})
    assert Metadata.model_validate(obj.model_dump()) == obj


def test_metadata_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Metadata.model_validate({"unknown_key": True})


# ---------------------------------------------------------------------------
# AblationConfig
# ---------------------------------------------------------------------------


def test_ablation_config_defaults() -> None:
    obj = AblationConfig.model_validate({})
    assert obj.subgroup_protocol == RulesProfile.full


def test_ablation_config_round_trip() -> None:
    obj = AblationConfig(subgroup_protocol=RulesProfile.naive)
    assert AblationConfig.model_validate(obj.model_dump()) == obj


def test_ablation_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AblationConfig.model_validate({"subgroup_protocol": "full", "extra": 1})


# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------


def test_analysis_config_defaults() -> None:
    obj = AnalysisConfig.model_validate(_analysis_config())
    assert obj.seed == 0
    assert obj.budget_s == 300.0
    assert obj.primary_metric == PrimaryMetric.macro_f1
    assert obj.run_experiments is True
    assert obj.rules_profile == RulesProfile.full
    assert obj.explain is True


def test_analysis_config_round_trip() -> None:
    obj = AnalysisConfig.model_validate(_analysis_config())
    assert AnalysisConfig.model_validate(obj.model_dump()) == obj


def test_analysis_config_json_round_trip() -> None:
    obj = AnalysisConfig.model_validate(_analysis_config())
    assert AnalysisConfig.model_validate_json(obj.model_dump_json()) == obj


def test_analysis_config_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisConfig.model_validate({**_analysis_config(), "mystery": 99})


# ---------------------------------------------------------------------------
# MetricResult
# ---------------------------------------------------------------------------


def test_metric_result_round_trip() -> None:
    obj = MetricResult.model_validate(_metric_result())
    assert MetricResult.model_validate(obj.model_dump()) == obj


def test_metric_result_with_ci() -> None:
    obj = MetricResult.model_validate({**_metric_result(), "ci_low": 0.70, "ci_high": 0.80})
    assert obj.ci_low == 0.70
    assert obj.ci_high == 0.80


def test_metric_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        MetricResult.model_validate({**_metric_result(), "bad": 1})


# ---------------------------------------------------------------------------
# Evidence
# ---------------------------------------------------------------------------


def test_evidence_round_trip() -> None:
    obj = Evidence.model_validate(_evidence())
    assert Evidence.model_validate(obj.model_dump()) == obj


def test_evidence_with_ci() -> None:
    obj = Evidence.model_validate({**_evidence(), "ci_low": 0.10, "ci_high": 0.28})
    assert obj.ci_low == 0.10


def test_evidence_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Evidence.model_validate({**_evidence(), "nope": 0})


# ---------------------------------------------------------------------------
# ExperimentSpec
# ---------------------------------------------------------------------------


def test_experiment_spec_round_trip() -> None:
    obj = ExperimentSpec.model_validate(_experiment_spec())
    assert ExperimentSpec.model_validate(obj.model_dump()) == obj


def test_experiment_spec_defaults() -> None:
    obj = ExperimentSpec.model_validate(_experiment_spec())
    assert obj.n_boot == 1000
    assert obj.repeats == 10
    assert obj.eval_on == ["holdout"]


def test_experiment_spec_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentSpec.model_validate({**_experiment_spec(), "surprise": True})


# ---------------------------------------------------------------------------
# NBResult
# ---------------------------------------------------------------------------


def test_nb_result_round_trip() -> None:
    obj = NBResult.model_validate(_nb_result())
    assert NBResult.model_validate(obj.model_dump()) == obj


def test_nb_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        NBResult.model_validate({**_nb_result(), "extra": 1})


# ---------------------------------------------------------------------------
# ExperimentResult
# ---------------------------------------------------------------------------


def test_experiment_result_round_trip() -> None:
    obj = ExperimentResult.model_validate(_experiment_result())
    assert ExperimentResult.model_validate(obj.model_dump()) == obj


def test_experiment_result_with_nb() -> None:
    obj = ExperimentResult.model_validate({**_experiment_result(), "nb": _nb_result()})
    assert obj.nb is not None
    assert obj.nb.mean == 0.03


def test_experiment_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExperimentResult.model_validate({**_experiment_result(), "extra": 1})


# ---------------------------------------------------------------------------
# TriggerResult
# ---------------------------------------------------------------------------


def test_trigger_result_round_trip() -> None:
    obj = TriggerResult.model_validate(_trigger())
    assert TriggerResult.model_validate(obj.model_dump()) == obj


def test_trigger_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        TriggerResult.model_validate({**_trigger(), "extra": 1})


# ---------------------------------------------------------------------------
# EffectResult
# ---------------------------------------------------------------------------


def test_effect_result_round_trip() -> None:
    obj = EffectResult.model_validate(_effect())
    assert EffectResult.model_validate(obj.model_dump()) == obj


def test_effect_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        EffectResult.model_validate({**_effect(), "extra": 1})


# ---------------------------------------------------------------------------
# ExplanationOutput
# ---------------------------------------------------------------------------


def test_explanation_output_round_trip() -> None:
    obj = ExplanationOutput(source="template", text="Overfitting was detected.")
    assert ExplanationOutput.model_validate(obj.model_dump()) == obj


def test_explanation_output_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ExplanationOutput.model_validate({"source": "template", "text": "x", "bad": 1})


# ---------------------------------------------------------------------------
# Decision
# ---------------------------------------------------------------------------


def test_decision_round_trip() -> None:
    obj = Decision.model_validate(_decision())
    assert Decision.model_validate(obj.model_dump()) == obj


def test_decision_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        Decision.model_validate({**_decision(), "extra": 1})


# ---------------------------------------------------------------------------
# HypothesisResult
# ---------------------------------------------------------------------------


def test_hypothesis_result_round_trip() -> None:
    obj = HypothesisResult.model_validate(_hypothesis_result())
    assert HypothesisResult.model_validate(obj.model_dump()) == obj


def test_hypothesis_result_with_subtype_imbalance() -> None:
    obj = HypothesisResult.model_validate(
        {**_hypothesis_result(), "type": "imbalance", "subtype": "distortion"}
    )
    assert obj.subtype == "distortion"


def test_hypothesis_result_with_subtype_leakage() -> None:
    for st in ("duplicate", "identifier", "target_proxy"):
        obj = HypothesisResult.model_validate(
            {**_hypothesis_result(), "type": "leakage", "subtype": st}
        )
        assert obj.subtype == st


def test_hypothesis_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        HypothesisResult.model_validate({**_hypothesis_result(), "extra": 1})


# ---------------------------------------------------------------------------
# ColumnProfile
# ---------------------------------------------------------------------------


def test_column_profile_round_trip() -> None:
    obj = ColumnProfile(
        name="age", inferred_type="numeric", missing_frac=0.02, n_unique=50, unique_ratio=0.1
    )
    assert ColumnProfile.model_validate(obj.model_dump()) == obj


def test_column_profile_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ColumnProfile.model_validate(
            {
                "name": "x",
                "inferred_type": "numeric",
                "missing_frac": 0,
                "n_unique": 1,
                "unique_ratio": 0.5,
                "extra": True,
            }
        )


# ---------------------------------------------------------------------------
# DatasetProfile
# ---------------------------------------------------------------------------


def test_dataset_profile_round_trip() -> None:
    obj = DatasetProfile(n_rows=1000, n_cols=10, class_balance={"0": 600, "1": 400})
    assert DatasetProfile.model_validate(obj.model_dump()) == obj


def test_dataset_profile_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        DatasetProfile.model_validate(
            {"n_rows": 100, "n_cols": 5, "class_balance": {"0": 60, "1": 40}, "extra": 1}
        )


# ---------------------------------------------------------------------------
# ReferenceResult
# ---------------------------------------------------------------------------


def test_reference_result_round_trip() -> None:
    obj = ReferenceResult.model_validate(_reference_result())
    assert ReferenceResult.model_validate(obj.model_dump()) == obj


def test_reference_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        ReferenceResult.model_validate({**_reference_result(), "extra": 1})


# ---------------------------------------------------------------------------
# AssessmentResult
# ---------------------------------------------------------------------------


def test_assessment_result_round_trip() -> None:
    obj = AssessmentResult.model_validate(_assessment())
    assert AssessmentResult.model_validate(obj.model_dump()) == obj


def test_assessment_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AssessmentResult.model_validate({**_assessment(), "extra": 1})


# ---------------------------------------------------------------------------
# AnalysisResult
# ---------------------------------------------------------------------------


def test_analysis_result_round_trip() -> None:
    obj = AnalysisResult.model_validate(_analysis_result())
    assert AnalysisResult.model_validate(obj.model_dump()) == obj


def test_analysis_result_json_round_trip() -> None:
    obj = AnalysisResult.model_validate(_analysis_result())
    assert AnalysisResult.model_validate_json(obj.model_dump_json()) == obj


def test_analysis_result_schema_version() -> None:
    obj = AnalysisResult.model_validate(_analysis_result())
    assert obj.schema_version == SCHEMA_VERSION == "1.0"


def test_analysis_result_rejects_unknown_fields() -> None:
    with pytest.raises(ValidationError):
        AnalysisResult.model_validate({**_analysis_result(), "extra": 1})
