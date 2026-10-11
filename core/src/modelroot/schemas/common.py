"""Common enums and the schema version constant shared by all ModelRoot contracts.

All enum values are API contracts and must not be changed without a major
schema-version bump and migration note.
"""

from __future__ import annotations

from enum import Enum

# ---------------------------------------------------------------------------
# Schema versioning
# ---------------------------------------------------------------------------

SCHEMA_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class Algorithm(str, Enum):
    """Supported training algorithms.

    Values are the stable string identifiers used in ``ModelSpec.algorithm``
    and in every artifact that references a trained model.
    """

    logistic_regression = "logistic_regression"
    decision_tree = "decision_tree"
    random_forest = "random_forest"
    hist_gradient_boosting = "hist_gradient_boosting"


class PrimaryMetric(str, Enum):
    """Optimisation / evaluation metric used as the primary performance signal."""

    accuracy = "accuracy"
    balanced_accuracy = "balanced_accuracy"
    macro_f1 = "macro_f1"
    f1 = "f1"
    roc_auc = "roc_auc"


class HypothesisType(str, Enum):
    """The five supported failure hypotheses."""

    overfitting = "overfitting"
    imbalance = "imbalance"
    leakage = "leakage"
    subgroup = "subgroup"
    shift = "shift"


class Status(str, Enum):
    """Verdict assigned to a hypothesis after evaluating its evidence."""

    supported = "supported"
    weakly_supported = "weakly_supported"
    inconclusive = "inconclusive"
    rejected = "rejected"
    not_triggered = "not_triggered"
    not_applicable = "not_applicable"


class Severity(str, Enum):
    """Estimated impact magnitude of a supported or weakly-supported hypothesis."""

    high = "high"
    medium = "medium"
    low = "low"


class EvidenceKind(str, Enum):
    """Whether an evidence item comes from a passive diagnostic or a paired experiment."""

    diagnostic = "diagnostic"
    experiment = "experiment"


class SplitMode(str, Enum):
    """How the train/test split is supplied by the user."""

    single_file = "single_file"
    split_column = "split_column"
    test_file = "test_file"


class ExperimentStatus(str, Enum):
    """Terminal state of an individual experiment run."""

    completed = "completed"
    timeout = "timeout"
    error = "error"


class RulesProfile(str, Enum):
    """Decision-rule strictness profile.

    ``full`` applies the complete multi-criteria protocol.
    ``naive`` is an ablation that uses only point-estimate Δ ≥ δ_min.
    """

    full = "full"
    naive = "naive"
