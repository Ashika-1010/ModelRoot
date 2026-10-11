"""Tests for all ModelRoot enums.

Verifies that:
- Every enum is importable from ``modelroot.schemas``.
- Every enum member has the exact string value specified by the contract.
- No accidental member additions or removals.

These tests are the first line of defence against silent renames or
re-orderings that would break external consumers of the API.
"""

from __future__ import annotations

import pytest

from modelroot.schemas import (
    Algorithm,
    EvidenceKind,
    ExperimentStatus,
    HypothesisType,
    PrimaryMetric,
    RulesProfile,
    Severity,
    SplitMode,
    Status,
)

# ---------------------------------------------------------------------------
# Algorithm
# ---------------------------------------------------------------------------


def test_algorithm_values() -> None:
    assert Algorithm.logistic_regression.value == "logistic_regression"
    assert Algorithm.decision_tree.value == "decision_tree"
    assert Algorithm.random_forest.value == "random_forest"
    assert Algorithm.hist_gradient_boosting.value == "hist_gradient_boosting"


def test_algorithm_members() -> None:
    names = {m.name for m in Algorithm}
    assert names == {
        "logistic_regression",
        "decision_tree",
        "random_forest",
        "hist_gradient_boosting",
    }


# ---------------------------------------------------------------------------
# PrimaryMetric
# ---------------------------------------------------------------------------


def test_primary_metric_values() -> None:
    assert PrimaryMetric.accuracy.value == "accuracy"
    assert PrimaryMetric.balanced_accuracy.value == "balanced_accuracy"
    assert PrimaryMetric.macro_f1.value == "macro_f1"
    assert PrimaryMetric.f1.value == "f1"
    assert PrimaryMetric.roc_auc.value == "roc_auc"


def test_primary_metric_members() -> None:
    names = {m.name for m in PrimaryMetric}
    assert names == {"accuracy", "balanced_accuracy", "macro_f1", "f1", "roc_auc"}


# ---------------------------------------------------------------------------
# HypothesisType
# ---------------------------------------------------------------------------


def test_hypothesis_type_values() -> None:
    assert HypothesisType.overfitting.value == "overfitting"
    assert HypothesisType.imbalance.value == "imbalance"
    assert HypothesisType.leakage.value == "leakage"
    assert HypothesisType.subgroup.value == "subgroup"
    assert HypothesisType.shift.value == "shift"


def test_hypothesis_type_members() -> None:
    names = {m.name for m in HypothesisType}
    assert names == {"overfitting", "imbalance", "leakage", "subgroup", "shift"}


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------


def test_status_values() -> None:
    assert Status.supported.value == "supported"
    assert Status.weakly_supported.value == "weakly_supported"
    assert Status.inconclusive.value == "inconclusive"
    assert Status.rejected.value == "rejected"
    assert Status.not_triggered.value == "not_triggered"
    assert Status.not_applicable.value == "not_applicable"


def test_status_members() -> None:
    names = {m.name for m in Status}
    assert names == {
        "supported",
        "weakly_supported",
        "inconclusive",
        "rejected",
        "not_triggered",
        "not_applicable",
    }


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


def test_severity_values() -> None:
    assert Severity.high.value == "high"
    assert Severity.medium.value == "medium"
    assert Severity.low.value == "low"


def test_severity_members() -> None:
    assert {m.name for m in Severity} == {"high", "medium", "low"}


# ---------------------------------------------------------------------------
# EvidenceKind
# ---------------------------------------------------------------------------


def test_evidence_kind_values() -> None:
    assert EvidenceKind.diagnostic.value == "diagnostic"
    assert EvidenceKind.experiment.value == "experiment"


def test_evidence_kind_members() -> None:
    assert {m.name for m in EvidenceKind} == {"diagnostic", "experiment"}


# ---------------------------------------------------------------------------
# SplitMode
# ---------------------------------------------------------------------------


def test_split_mode_values() -> None:
    assert SplitMode.single_file.value == "single_file"
    assert SplitMode.split_column.value == "split_column"
    assert SplitMode.test_file.value == "test_file"


def test_split_mode_members() -> None:
    assert {m.name for m in SplitMode} == {"single_file", "split_column", "test_file"}


# ---------------------------------------------------------------------------
# ExperimentStatus
# ---------------------------------------------------------------------------


def test_experiment_status_values() -> None:
    assert ExperimentStatus.completed.value == "completed"
    assert ExperimentStatus.timeout.value == "timeout"
    assert ExperimentStatus.error.value == "error"


def test_experiment_status_members() -> None:
    assert {m.name for m in ExperimentStatus} == {"completed", "timeout", "error"}


# ---------------------------------------------------------------------------
# RulesProfile
# ---------------------------------------------------------------------------


def test_rules_profile_values() -> None:
    assert RulesProfile.full.value == "full"
    assert RulesProfile.naive.value == "naive"


def test_rules_profile_members() -> None:
    assert {m.name for m in RulesProfile} == {"full", "naive"}


# ---------------------------------------------------------------------------
# Enums are str subclasses (needed for JSON serialisation and comparisons)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "enum_cls",
    [
        Algorithm,
        PrimaryMetric,
        HypothesisType,
        Status,
        Severity,
        EvidenceKind,
        SplitMode,
        ExperimentStatus,
        RulesProfile,
    ],
)
def test_enum_is_str_subclass(enum_cls: type) -> None:
    for member in enum_cls:
        assert isinstance(member, str), (
            f"{enum_cls.__name__}.{member.name} should be a str, got {type(member)}"
        )
