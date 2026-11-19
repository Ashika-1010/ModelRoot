"""Tests for benchmark scoring and report generation (M12).

Validates scoring formulas, detection sets, adjusted-FP logic, H3 subtype separation,
edge cases, and bootstrap determinism against hand-crafted results.
"""

from __future__ import annotations

from modelroot_bench.scenario import (
    BenchmarkRecord,
    GroundTruth,
    InjectedFailure,
    ScenarioResult,
    ScenarioSpec,
)
from modelroot_bench.scoring import score


def _make_record(
    scenario_id: str,
    scenario_class: str,
    injected: list[str],
    diagnosed_statuses: dict[str, str],
    control_statuses: dict[str, str] | None = None,
    no_effect: bool = False,
    top1: str | None = None,
    effects: dict[str, float] | None = None,
    runtime_s: float = 1.0,
) -> BenchmarkRecord:
    spec = ScenarioSpec(
        id=scenario_id,
        dataset="synth",
        model_spec={"algorithm": "random_forest", "params": {}},
        scenario_class=scenario_class,  # type: ignore[arg-type]
        injections=[],
        seed=0,
        role="dev",
    )
    gt = GroundTruth(
        injected=[InjectedFailure(label=lbl, severity="S1") for lbl in injected],
        impact=0.01 if no_effect else 0.05,
        no_effect=no_effect,
    )
    res = ScenarioResult(
        scenario_id=scenario_id,
        statuses=diagnosed_statuses,
        top1=top1,
        runtime_s=runtime_s,
        analysis_hash="hash_" + scenario_id,
        control_statuses=control_statuses or {},
        effects=effects or {},
    )
    return BenchmarkRecord(spec=spec, ground_truth=gt, result=res)


def test_hand_crafted_confusion_matrix() -> None:
    """Validate 10 hand-built cases with known exact arithmetic."""
    recs: list[BenchmarkRecord] = [
        # Case 1: Injected, supported (TP lenient, TP strict)
        _make_record("c1", "single_failure", ["overfitting"], {"overfitting": "supported"}),
        # Case 2: Injected, weakly_supported (TP lenient, FN strict)
        _make_record("c2", "single_failure", ["overfitting"], {"overfitting": "weakly_supported"}),
        # Case 3: Injected, rejected (FN lenient, FN strict)
        _make_record("c3", "single_failure", ["overfitting"], {"overfitting": "rejected"}),
        # Case 4: Injected with no_effect=True, supported (FP for precision, excluded from recall)
        _make_record(
            "c4",
            "single_failure",
            ["overfitting"],
            {"overfitting": "supported"},
            no_effect=True,
        ),
        # Case 5: Injected with no_effect=True, not_triggered (excluded from recall denominator)
        _make_record(
            "c5",
            "single_failure",
            ["overfitting"],
            {"overfitting": "not_triggered"},
            no_effect=True,
        ),
        # Case 6: Clean control, diagnosed supported (FP raw & adjusted)
        _make_record("c6", "clean_control", [], {"overfitting": "supported"}),
        # Case 7: Clean control, natural in control (FP raw, discounted in adjusted)
        _make_record(
            "c7",
            "clean_control",
            [],
            {"overfitting": "supported"},
            control_statuses={"overfitting": "supported"},
        ),
        # Case 8: Clean control, not_triggered (TN)
        _make_record("c8", "clean_control", [], {"overfitting": "not_triggered"}),
        # Case 9: Clean control, rejected (TN)
        _make_record("c9", "clean_control", [], {"overfitting": "rejected"}),
        # Case 10: Clean control, inconclusive (TN)
        _make_record("c10", "clean_control", [], {"overfitting": "inconclusive"}),
    ]

    report = score(recs, bootstrap=False)
    m_len = report.lenient_metrics["overfitting"]
    m_str = report.strict_metrics["overfitting"]

    # Lenient calculations:
    # TP = c1 (supp), c2 (weak) = 2 (c4 is no_effect -> counted as FP, not TP)
    assert m_len.tp == 2
    # Recall denominator = c1, c2, c3 = 3 (c4, c5 excluded by no_effect)
    # Lenient detected in denominator = c1, c2 = 2
    # Recall = 2 / 3
    assert m_len.recall is not None
    assert abs(m_len.recall - 2.0 / 3.0) < 1e-9

    # Raw FP = c4 (no_effect detected), c6, c7 = 3
    assert m_len.fp == 3
    # Raw Precision = 2 / (2 + 3) = 0.40
    assert m_len.precision is not None
    assert abs(m_len.precision - 0.40) < 1e-9

    # Adjusted FP = c4, c6 = 2 (c7 was natural in control)
    assert m_len.adjusted_fp == 2
    # Adjusted Precision = 2 / (2 + 2) = 0.50
    assert m_len.adjusted_precision is not None
    assert abs(m_len.adjusted_precision - 0.50) < 1e-9

    # F1 = 2 * (2/5) * (2/3) / (2/5 + 2/3) = 0.50
    assert m_len.f1 is not None
    assert abs(m_len.f1 - 0.50) < 1e-9

    # Strict calculations:
    # Strict TP = c1 = 1 (c4 is no_effect -> counted as FP, not TP)
    assert m_str.tp == 1
    # Strict detected in valid recall denominator = c1 = 1
    # Strict Recall = 1 / 3
    assert m_str.recall is not None
    assert abs(m_str.recall - 1.0 / 3.0) < 1e-9
    # Strict Precision = 1 / (1 + 3) = 0.25
    assert m_str.precision is not None
    assert abs(m_str.precision - 0.25) < 1e-9
    # Strict Adjusted Precision = 1 / (1 + 2) = 1 / 3
    assert m_str.adjusted_precision is not None
    assert abs(m_str.adjusted_precision - 1.0 / 3.0) < 1e-9


def test_edge_case_no_positives_recall_is_none() -> None:
    """When no failures are injected, recall must be None (not 0.0)."""
    recs = [
        _make_record("c1", "clean_control", [], {"overfitting": "not_triggered"}),
        _make_record("c2", "clean_control", [], {"overfitting": "rejected"}),
    ]
    report = score(recs, bootstrap=False)
    m = report.lenient_metrics["overfitting"]
    assert m.recall is None
    assert m.precision is None
    assert m.f1 is None


def test_edge_case_no_detections_precision_is_none() -> None:
    """When nothing is detected, precision must be None (not 0.0)."""
    recs = [
        _make_record("c1", "single_failure", ["overfitting"], {"overfitting": "rejected"}),
        _make_record("c2", "clean_control", [], {"overfitting": "not_triggered"}),
    ]
    report = score(recs, bootstrap=False)
    m = report.lenient_metrics["overfitting"]
    assert m.tp == 0
    assert m.fp == 0
    assert m.fn == 1
    assert m.precision is None
    assert m.recall == 0.0
    assert m.f1 is None


def test_h3_subtypes_separate_scoring_and_aggregation() -> None:
    """Validate Fix F4: H3 subtypes are scored separately AND aggregated to type-level."""
    recs = [
        # Subtype duplicate injected and detected
        _make_record(
            "c1",
            "single_failure",
            ["leakage:duplicate"],
            {
                "leakage:duplicate": "supported",
                "leakage:identifier": "not_triggered",
            },
        ),
        # Subtype identifier injected but not detected
        _make_record(
            "c2",
            "single_failure",
            ["leakage:identifier"],
            {
                "leakage:duplicate": "not_triggered",
                "leakage:identifier": "rejected",
            },
        ),
    ]
    report = score(recs, bootstrap=False)

    # Subtype duplicate: 1 TP, 0 FN -> recall = 1.0
    dup_m = report.lenient_metrics["leakage:duplicate"]
    assert dup_m.tp == 1
    assert dup_m.fn == 0
    assert dup_m.recall == 1.0

    # Subtype identifier: 0 TP, 1 FN -> recall = 0.0
    id_m = report.lenient_metrics["leakage:identifier"]
    assert id_m.tp == 0
    assert id_m.fn == 1
    assert id_m.recall == 0.0

    # Type-level leakage: c1 aggregated to supported, c2 to rejected -> 1 TP, 1 FN -> recall = 0.5
    leak_m = report.lenient_metrics["leakage"]
    assert leak_m.tp == 1
    assert leak_m.fn == 1
    assert leak_m.recall == 0.5


def test_attribution_top1_and_mrr() -> None:
    """Validate attribution top-1 and MRR formulas on single failures."""
    recs = [
        # Case A: top1 matches (rank 1 -> RR = 1.0)
        _make_record(
            "c1",
            "single_failure",
            ["overfitting"],
            {"overfitting": "supported", "imbalance": "weakly_supported"},
            top1="overfitting",
            effects={"overfitting": 0.05, "imbalance": 0.02},
        ),
        # Case B: top1 does not match; true failure ranked second (RR = 0.5)
        _make_record(
            "c2",
            "single_failure",
            ["imbalance"],
            {"overfitting": "supported", "imbalance": "weakly_supported"},
            top1="overfitting",
            effects={"overfitting": 0.05, "imbalance": 0.02},
        ),
        # Case C: true failure not diagnosed supported/weakly (RR = 0.0)
        _make_record(
            "c3",
            "single_failure",
            ["shift"],
            {"overfitting": "supported", "shift": "rejected"},
            top1="overfitting",
            effects={"overfitting": 0.05, "shift": 0.0},
        ),
    ]
    report = score(recs, bootstrap=False)
    attr = report.attribution

    assert attr.n_single_failure == 3
    # Top 1 correct for c1 only
    assert attr.top1_accuracy is not None
    assert abs(attr.top1_accuracy - 1.0 / 3.0) < 1e-9
    # MRR = (1.0 + 0.5 + 0.0) / 3 = 0.50
    assert attr.mrr is not None
    assert abs(attr.mrr - 0.50) < 1e-9


def test_multi_failure_jaccard() -> None:
    """Validate Jaccard calculation on multi-failure scenarios."""
    # G = {overfitting, imbalance}
    # Detected strict = {overfitting, leakage}
    # Intersection = 1, Union = 3 -> Jaccard = 1/3
    rec = _make_record(
        "mf1",
        "multi_failure",
        ["overfitting", "imbalance"],
        {
            "overfitting": "supported",
            "imbalance": "weakly_supported",  # not strict
            "leakage": "supported",
        },
    )
    report = score([rec], bootstrap=False)
    assert report.attribution.n_multi_failure == 1
    assert report.attribution.jaccard_strict is not None
    assert abs(report.attribution.jaccard_strict - 1.0 / 3.0) < 1e-9


def test_reliability_table_counts() -> None:
    """Validate status calibration table counts."""
    recs = [
        _make_record("c1", "single_failure", ["overfitting"], {"overfitting": "supported"}),
        _make_record("c2", "clean_control", [], {"overfitting": "supported"}),
        _make_record("c3", "single_failure", ["overfitting"], {"overfitting": "rejected"}),
    ]
    report = score(recs, bootstrap=False)
    supp_entry = report.reliability_table["supported"]
    assert supp_entry.total_assigned == 2
    assert supp_entry.truly_injected == 1
    assert supp_entry.precision == 0.5


def test_bootstrap_ci_determinism() -> None:
    """Validate deterministic bootstrap confidence intervals."""
    recs: list[BenchmarkRecord] = [
        _make_record(f"c{i}", "single_failure", ["overfitting"], {"overfitting": "supported"})
        for i in range(5)
    ] + [
        _make_record(f"cc{i}", "clean_control", [], {"overfitting": "not_triggered"})
        for i in range(5)
    ]
    r1 = score(recs, bootstrap=True, n_boot=100, seed=123)
    r2 = score(recs, bootstrap=True, n_boot=100, seed=123)

    assert r1.bootstrap_ci is not None
    assert r2.bootstrap_ci is not None
    assert r1.bootstrap_ci.macro_precision == r2.bootstrap_ci.macro_precision
    assert r1.bootstrap_ci.macro_recall == r2.bootstrap_ci.macro_recall


def test_score_report_summary_table() -> None:
    """Verify summary table formatting does not raise."""
    recs = [
        _make_record("c1", "single_failure", ["overfitting"], {"overfitting": "supported"}),
        _make_record("c2", "clean_control", [], {"overfitting": "not_triggered"}),
    ]
    report = score(recs, bootstrap=False)
    table_str = report.summary_table()
    assert "ModelRoot Benchmark Score Report" in table_str
    assert "Macro Precision" in table_str


def test_no_effect_precision_and_recall_semantics() -> None:
    """Verify scoring semantics for no_effect:
    - no_effect=False and detected -> TP, in recall
    - no_effect=False and not detected -> FN
    - no_effect=True and detected -> FP, not TP (excluded from recall)
    - no_effect=True and not detected -> excluded from both precision and recall
    - no_effect=True and detected, but natural in control -> FP raw, discounted in adjusted FP
    """
    recs: list[BenchmarkRecord] = [
        # 1. Injected, meaningful impact, detected -> TP
        _make_record(
            "r1", "single_failure", ["overfitting"], {"overfitting": "supported"}, no_effect=False
        ),
        # 2. Injected, meaningful impact, not detected -> FN
        _make_record(
            "r2", "single_failure", ["overfitting"], {"overfitting": "rejected"}, no_effect=False
        ),
        # 3. Injected, negligible impact (no_effect=True), detected -> FP (over-diagnosis)
        _make_record(
            "r3", "single_failure", ["overfitting"], {"overfitting": "supported"}, no_effect=True
        ),
        # 4. Injected, negligible impact (no_effect=True), not detected -> Excluded from P & R
        _make_record(
            "r4",
            "single_failure",
            ["overfitting"],
            {"overfitting": "not_triggered"},
            no_effect=True,
        ),
        # 5. Injected, negligible impact (no_effect=True), detected, but natural in control
        # -> raw FP, discounted in adj_fp
        _make_record(
            "r5",
            "single_failure",
            ["overfitting"],
            {"overfitting": "supported"},
            control_statuses={"overfitting": "supported"},
            no_effect=True,
        ),
    ]

    report = score(recs, bootstrap=False)
    m = report.strict_metrics["overfitting"]

    # True Positives: r1 only
    assert m.tp == 1
    # Recall denominator: r1, r2 (r3, r4, r5 excluded by no_effect=True)
    # Detected in valid recall denominator: r1 = 1
    # FN: r2 = 1
    assert m.fn == 1
    assert m.recall == 0.5  # 1 / (1 + 1)

    # Raw False Positives: r3, r5
    assert m.fp == 2
    # Precision: tp / (tp + fp) = 1 / (1 + 2) = 1/3
    assert m.precision is not None
    assert abs(m.precision - 1.0 / 3.0) < 1e-9

    # Adjusted FP: r3 only (r5 was natural in control)
    assert m.adjusted_fp == 1
    # Adjusted Precision: tp / (tp + adj_fp) = 1 / (1 + 1) = 0.5
    assert m.adjusted_precision == 0.5
