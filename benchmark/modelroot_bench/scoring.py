"""Benchmark scoring, ground-truth evaluation, and report generation (M12).

Implements the multi-dimensional scoring framework specified in §6.4 of the
ModelRoot Master Plan:
- Lenient (status >= weakly_supported) and strict (status == supported) detection sets.
- Type-level aggregation and separate H3 subtype evaluation (Fix F4).
- Exclusion of no_effect (impact < 0.02) from recall denominators.
- Per-hypothesis and macro Precision, Recall, F1.
- Family-wise and per-hypothesis False Positive Rate (FPR) across scenario classes
  (clean_control, distractor, hard_negative).
- Adjusted FP logic discounting natural flags observed in matching clean controls.
- Attribution metrics: Top-1 accuracy and Mean Reciprocal Rank (MRR) on single-failure
  scenarios; Jaccard similarity on multi-failure scenarios.
- Status calibration (reliability table): fraction truly injected per status.
- Severity-response curve.
- Timing summary.
- Deterministic bootstrap confidence intervals over scenarios.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from modelroot_bench.scenario import BenchmarkRecord, ScenarioResult

# Status ranking order: supported (5) > weakly_supported (4) > inconclusive (3)
# > rejected (2) > not_triggered (1) > not_applicable (0)
STATUS_SEVERITY_ORDER: dict[str, int] = {
    "supported": 5,
    "weakly_supported": 4,
    "inconclusive": 3,
    "rejected": 2,
    "not_triggered": 1,
    "not_applicable": 0,
}

# Standard hypothesis canonical names
H3_SUBTYPES = ("leakage:duplicate", "leakage:identifier", "leakage:target_proxy")


class MetricScore(BaseModel):
    """Container for precision, recall, and F1."""

    model_config = ConfigDict(extra="forbid")

    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    tp: int = 0
    fp: int = 0
    fn: int = 0
    adjusted_fp: int = 0
    adjusted_precision: float | None = None
    adjusted_f1: float | None = None


class ClassFPR(BaseModel):
    """False Positive Rate breakdown for a scenario class."""

    model_config = ConfigDict(extra="forbid")

    n_scenarios: int = 0
    n_false_positive_scenarios: int = 0
    raw_fpr: float = 0.0
    n_adjusted_fp_scenarios: int = 0
    adjusted_fpr: float = 0.0


class AttributionMetrics(BaseModel):
    """Attribution metrics on single and multi-failure scenarios."""

    model_config = ConfigDict(extra="forbid")

    n_single_failure: int = 0
    top1_accuracy: float | None = None
    mrr: float | None = None
    n_multi_failure: int = 0
    jaccard_strict: float | None = None


class ReliabilityEntry(BaseModel):
    """Reliability entry for one status in the calibration table."""

    model_config = ConfigDict(extra="forbid")

    status: str
    total_assigned: int = 0
    truly_injected: int = 0
    precision: float | None = None


class SeverityResponseEntry(BaseModel):
    """Detection rates for one severity level."""

    model_config = ConfigDict(extra="forbid")

    severity: str
    n_scenarios: int = 0
    lenient_detection_rate: float = 0.0
    strict_detection_rate: float = 0.0


class TimingSummary(BaseModel):
    """Summary of scenario execution wall-clock runtimes."""

    model_config = ConfigDict(extra="forbid")

    mean_s: float = 0.0
    median_s: float = 0.0
    min_s: float = 0.0
    max_s: float = 0.0
    total_s: float = 0.0


class BootstrapCI(BaseModel):
    """Bootstrap confidence intervals (95% percentile)."""

    model_config = ConfigDict(extra="forbid")

    n_scenarios: int
    n_boot: int
    macro_precision: tuple[float, float] | None = None
    macro_recall: tuple[float, float] | None = None
    macro_f1: tuple[float, float] | None = None
    clean_control_fpr: tuple[float, float] | None = None


class ScoreReport(BaseModel):
    """Comprehensive benchmark scoring report."""

    model_config = ConfigDict(extra="forbid")

    n_scenarios: int
    lenient_metrics: dict[str, MetricScore] = Field(default_factory=dict)
    strict_metrics: dict[str, MetricScore] = Field(default_factory=dict)
    macro_lenient: MetricScore = Field(default_factory=MetricScore)
    macro_strict: MetricScore = Field(default_factory=MetricScore)
    fpr_by_class: dict[str, ClassFPR] = Field(default_factory=dict)
    attribution: AttributionMetrics = Field(default_factory=AttributionMetrics)
    reliability_table: dict[str, ReliabilityEntry] = Field(default_factory=dict)
    severity_response: dict[str, SeverityResponseEntry] = Field(default_factory=dict)
    timing: TimingSummary = Field(default_factory=TimingSummary)
    bootstrap_ci: BootstrapCI | None = None

    def summary_table(self) -> str:
        """Render a concise plain-text summary table."""
        lines = [
            f"ModelRoot Benchmark Score Report (n={self.n_scenarios})",
            "=" * 60,
            f"{'Metric':<25} {'Lenient':<15} {'Strict':<15}",
            "-" * 60,
        ]

        def _fmt(val: float | None) -> str:
            return f"{val:.4f}" if val is not None else "None"

        len_p = _fmt(self.macro_lenient.precision)
        str_p = _fmt(self.macro_strict.precision)
        lines.append(f"{'Macro Precision':<25} {len_p:<15} {str_p:<15}")

        len_r = _fmt(self.macro_lenient.recall)
        str_r = _fmt(self.macro_strict.recall)
        lines.append(f"{'Macro Recall':<25} {len_r:<15} {str_r:<15}")

        len_f = _fmt(self.macro_lenient.f1)
        str_f = _fmt(self.macro_strict.f1)
        lines.append(f"{'Macro F1':<25} {len_f:<15} {str_f:<15}")

        len_ap = _fmt(self.macro_lenient.adjusted_precision)
        str_ap = _fmt(self.macro_strict.adjusted_precision)
        lines.append(f"{'Adjusted Precision':<25} {len_ap:<15} {str_ap:<15}")

        lines.append("-" * 60)
        clean_fpr = self.fpr_by_class.get("clean_control")
        if clean_fpr:
            lines.append(
                f"{'Clean Control FPR':<25} {clean_fpr.raw_fpr:.4f} "
                f"(adj: {clean_fpr.adjusted_fpr:.4f})"
            )
        if self.attribution.top1_accuracy is not None:
            lines.append(f"{'Top-1 Accuracy':<25} {self.attribution.top1_accuracy:.4f}")
        if self.attribution.mrr is not None:
            lines.append(f"{'Attribution MRR':<25} {self.attribution.mrr:.4f}")
        if self.attribution.jaccard_strict is not None:
            lines.append(f"{'Multi-failure Jaccard':<25} {self.attribution.jaccard_strict:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)


def _canonicalize_label(label: str) -> str:
    """Normalize a hypothesis label."""
    return label.strip().lower()


def _is_subtype(label: str) -> bool:
    """Check if label is an explicit subtype claim (e.g. leakage:duplicate)."""
    return ":" in label


def _type_level_name(label: str) -> str:
    """Extract type-level name (e.g. 'leakage:duplicate' -> 'leakage')."""
    return label.split(":", 1)[0]


def _status_order(status: str) -> int:
    return STATUS_SEVERITY_ORDER.get(status, -1)


def _best_status(statuses: list[str]) -> str:
    if not statuses:
        return "not_triggered"
    return max(statuses, key=_status_order)


def _aggregate_statuses(raw_statuses: dict[str, str]) -> dict[str, str]:
    """Aggregate sub-claim statuses to type-level per Fix F4.

    Type-level status = best status among sub-claims for H1/H2/H4/H5.
    H3 subtypes are retained individually AND aggregated to 'leakage'.
    """
    aggregated: dict[str, str] = dict(raw_statuses)
    by_type: dict[str, list[str]] = defaultdict(list)

    for k, s in raw_statuses.items():
        t = _type_level_name(k)
        by_type[t].append(s)

    for t, stat_list in by_type.items():
        aggregated[t] = _best_status(stat_list)

    return aggregated


def _is_detected(status: str, strict: bool) -> bool:
    """Check whether a status counts as a detection."""
    if strict:
        return status == "supported"
    return status in ("supported", "weakly_supported")


def _compute_p_r_f1(
    tp: int,
    tp_recall: int,
    fp: int,
    fn: int,
    adj_fp: int,
) -> tuple[float | None, float | None, float | None, float | None, float | None]:
    """Compute precision, recall, f1, adjusted_precision, adjusted_f1."""
    precision: float | None = (tp / (tp + fp)) if (tp + fp) > 0 else None
    recall: float | None = (tp_recall / (tp_recall + fn)) if (tp_recall + fn) > 0 else None
    f1: float | None = None
    if precision is not None and recall is not None:
        f1 = (2.0 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    adj_prec: float | None = (tp / (tp + adj_fp)) if (tp + adj_fp) > 0 else None
    adj_f1: float | None = None
    if adj_prec is not None and recall is not None:
        adj_f1 = (2.0 * adj_prec * recall / (adj_prec + recall)) if (adj_prec + recall) > 0 else 0.0

    return precision, recall, f1, adj_prec, adj_f1


def _rank_hypotheses(
    statuses: dict[str, str],
    effects: dict[str, float],
    ranking: list[str] | None = None,
) -> list[str]:
    """Rank hypotheses by status order desc, then effect delta desc, then label."""
    if ranking:
        return list(ranking)

    def _key(lbl: str) -> tuple[int, float, str]:
        st = statuses.get(lbl, "not_applicable")
        st_val = _status_order(st)
        eff = effects.get(lbl, 0.0)
        return (-st_val, -eff, lbl)

    return sorted(statuses.keys(), key=_key)


def score(
    records: list[BenchmarkRecord],
    bootstrap: bool = True,
    n_boot: int = 1000,
    seed: int = 42,
) -> ScoreReport:
    """Score a collection of benchmark records according to §6.4.

    Args:
        records: List of evaluated :class:`BenchmarkRecord` objects.
        bootstrap: Whether to compute bootstrap confidence intervals.
        n_boot: Number of bootstrap iterations.
        seed: Deterministic RNG seed for bootstrap resampling.

    Returns:
        A complete, structured :class:`ScoreReport`.
    """
    n_scenarios = len(records)
    if n_scenarios == 0:
        return ScoreReport(
            n_scenarios=0,
            timing=TimingSummary(),
        )

    # 1. Collect all evaluated targets (hypotheses and H3 subtypes)
    all_targets: set[str] = set()
    for rec in records:
        for inj in rec.ground_truth.injected:
            lbl = _canonicalize_label(inj.label)
            all_targets.add(lbl)
            if _is_subtype(lbl):
                all_targets.add(_type_level_name(lbl))
        if rec.result:
            for lbl in rec.result.statuses:
                c_lbl = _canonicalize_label(lbl)
                all_targets.add(c_lbl)
                if _is_subtype(c_lbl):
                    all_targets.add(_type_level_name(c_lbl))

    # Fallback to standard if empty
    if not all_targets:
        all_targets.add("overfitting")

    # 2. Compute per-hypothesis metrics for lenient and strict
    def _evaluate_detection(strict: bool) -> tuple[dict[str, MetricScore], MetricScore]:
        per_target: dict[str, MetricScore] = {}

        for target in sorted(all_targets):
            tp = 0
            tp_recall = 0
            fp = 0
            fn = 0
            adj_fp = 0

            for rec in records:
                gt = rec.ground_truth
                res = rec.result or ScenarioResult(scenario_id=rec.spec.id)

                # Check if target was injected
                injected_labels = [_canonicalize_label(i.label) for i in gt.injected]
                is_injected = (target in injected_labels) or any(
                    _type_level_name(i) == target for i in injected_labels
                )

                # Determine diagnosed status for this target
                statuses = _aggregate_statuses(res.statuses)
                st = statuses.get(target, "not_triggered")
                detected = _is_detected(st, strict=strict)

                # Check clean control status for adjusted FP
                ctrl_statuses = _aggregate_statuses(res.control_statuses)
                ctrl_st = ctrl_statuses.get(target, "not_triggered")
                natural_in_control = _is_detected(ctrl_st, strict=strict)

                if is_injected:
                    if gt.no_effect:
                        # Injected failure with negligible impact (<0.02)
                        # Excluded from recall; if detected, counts as FP (over-diagnosis)
                        if detected:
                            fp += 1
                            if not natural_in_control:
                                adj_fp += 1
                    else:
                        if detected:
                            tp += 1
                            tp_recall += 1
                        else:
                            fn += 1
                else:
                    if detected:
                        fp += 1
                        if not natural_in_control:
                            adj_fp += 1

            p, r, f, ap, af = _compute_p_r_f1(tp, tp_recall, fp, fn, adj_fp)
            per_target[target] = MetricScore(
                precision=p,
                recall=r,
                f1=f,
                tp=tp,
                fp=fp,
                fn=fn,
                adjusted_fp=adj_fp,
                adjusted_precision=ap,
                adjusted_f1=af,
            )

        # Macro aggregation
        p_vals = [m.precision for m in per_target.values() if m.precision is not None]
        r_vals = [m.recall for m in per_target.values() if m.recall is not None]
        f_vals = [m.f1 for m in per_target.values() if m.f1 is not None]
        ap_vals = [
            m.adjusted_precision for m in per_target.values() if m.adjusted_precision is not None
        ]
        af_vals = [m.adjusted_f1 for m in per_target.values() if m.adjusted_f1 is not None]

        macro = MetricScore(
            precision=float(np.mean(p_vals)) if p_vals else None,
            recall=float(np.mean(r_vals)) if r_vals else None,
            f1=float(np.mean(f_vals)) if f_vals else None,
            adjusted_precision=float(np.mean(ap_vals)) if ap_vals else None,
            adjusted_f1=float(np.mean(af_vals)) if af_vals else None,
            tp=sum(m.tp for m in per_target.values()),
            fp=sum(m.fp for m in per_target.values()),
            fn=sum(m.fn for m in per_target.values()),
            adjusted_fp=sum(m.adjusted_fp for m in per_target.values()),
        )
        return per_target, macro

    lenient_metrics, macro_lenient = _evaluate_detection(strict=False)
    strict_metrics, macro_strict = _evaluate_detection(strict=True)

    # 3. FPR by scenario class (clean_control, distractor, hard_negative)
    fpr_by_class: dict[str, ClassFPR] = {}
    classes = ("clean_control", "distractor", "hard_negative")
    for cls_name in classes:
        cls_records = [r for r in records if r.spec.scenario_class == cls_name]
        n_cls = len(cls_records)
        n_raw_fp = 0
        n_adj_fp = 0

        for rec in cls_records:
            res = rec.result or ScenarioResult(scenario_id=rec.spec.id)
            statuses = _aggregate_statuses(res.statuses)
            ctrl_statuses = _aggregate_statuses(res.control_statuses)

            # Check if any non-injected failure was detected leniently
            injected_set = {_canonicalize_label(i.label) for i in rec.ground_truth.injected} | {
                _type_level_name(_canonicalize_label(i.label)) for i in rec.ground_truth.injected
            }

            has_raw = False
            has_adj = False
            for k, st in statuses.items():
                if k not in injected_set and _is_detected(st, strict=False):
                    has_raw = True
                    ctrl_st = ctrl_statuses.get(k, "not_triggered")
                    if not _is_detected(ctrl_st, strict=False):
                        has_adj = True

            if has_raw:
                n_raw_fp += 1
            if has_adj:
                n_adj_fp += 1

        raw_rate = (n_raw_fp / n_cls) if n_cls > 0 else 0.0
        adj_rate = (n_adj_fp / n_cls) if n_cls > 0 else 0.0
        fpr_by_class[cls_name] = ClassFPR(
            n_scenarios=n_cls,
            n_false_positive_scenarios=n_raw_fp,
            raw_fpr=raw_rate,
            n_adjusted_fp_scenarios=n_adj_fp,
            adjusted_fpr=adj_rate,
        )

    # 4. Attribution: Top-1 Accuracy & MRR for single-failure; Jaccard for multi-failure
    single_failures = [r for r in records if r.spec.scenario_class == "single_failure"]
    n_single = len(single_failures)
    top1_correct = 0
    reciprocal_ranks: list[float] = []

    for rec in single_failures:
        if not rec.ground_truth.injected:
            continue
        true_label = _canonicalize_label(rec.ground_truth.injected[0].label)
        true_type = _type_level_name(true_label)
        res = rec.result or ScenarioResult(scenario_id=rec.spec.id)

        # Top-1 accuracy check
        if res.top1 is not None:
            c_top = _canonicalize_label(res.top1)
            if c_top == true_label or _type_level_name(c_top) == true_type:
                top1_correct += 1

        # MRR calculation
        statuses = _aggregate_statuses(res.statuses)
        ranked = _rank_hypotheses(statuses, res.effects, res.ranking)

        # Find position of true failure among ranked hypotheses
        rank_val: int | None = None
        for idx, h_lbl in enumerate(ranked, start=1):
            c_h = _canonicalize_label(h_lbl)
            if c_h == true_label or _type_level_name(c_h) == true_type:
                # Must be supported or weakly_supported to count towards MRR
                if _is_detected(statuses.get(h_lbl, "not_triggered"), strict=False):
                    rank_val = idx
                break

        if rank_val is not None:
            reciprocal_ranks.append(1.0 / rank_val)
        else:
            reciprocal_ranks.append(0.0)

    top1_acc = (top1_correct / n_single) if n_single > 0 else None
    mean_mrr = float(np.mean(reciprocal_ranks)) if reciprocal_ranks else None

    # Multi-failure Jaccard
    multi_failures = [r for r in records if r.spec.scenario_class == "multi_failure"]
    n_multi = len(multi_failures)
    jaccard_scores: list[float] = []

    for rec in multi_failures:
        res = rec.result or ScenarioResult(scenario_id=rec.spec.id)
        g_set = {_canonicalize_label(i.label) for i in rec.ground_truth.injected}
        s_star = {
            _canonicalize_label(k)
            for k, st in _aggregate_statuses(res.statuses).items()
            if _is_detected(st, strict=True)
        }
        union_len = len(g_set | s_star)
        jaccard = (len(g_set & s_star) / union_len) if union_len > 0 else 1.0
        jaccard_scores.append(jaccard)

    mean_jaccard = float(np.mean(jaccard_scores)) if jaccard_scores else None

    attribution = AttributionMetrics(
        n_single_failure=n_single,
        top1_accuracy=top1_acc,
        mrr=mean_mrr,
        n_multi_failure=n_multi,
        jaccard_strict=mean_jaccard,
    )

    # 5. Status Calibration (Reliability Table)
    reliability_table: dict[str, ReliabilityEntry] = {}
    for st_name in STATUS_SEVERITY_ORDER:
        assigned = 0
        truly_inj = 0
        for rec in records:
            res = rec.result or ScenarioResult(scenario_id=rec.spec.id)
            gt_injected = {_canonicalize_label(i.label) for i in rec.ground_truth.injected}
            gt_types = {_type_level_name(i) for i in gt_injected}

            for hyp, diagnosed_st in _aggregate_statuses(res.statuses).items():
                if diagnosed_st == st_name:
                    assigned += 1
                    c_h = _canonicalize_label(hyp)
                    if c_h in gt_injected or _type_level_name(c_h) in gt_types:
                        truly_inj += 1

        prec = (truly_inj / assigned) if assigned > 0 else None
        reliability_table[st_name] = ReliabilityEntry(
            status=st_name,
            total_assigned=assigned,
            truly_injected=truly_inj,
            precision=prec,
        )

    # 6. Severity-Response Curve
    severity_map: dict[str, list[tuple[bool, bool]]] = defaultdict(list)
    for rec in records:
        for inj in rec.ground_truth.injected:
            sev = inj.severity or "unspecified"
            target = _canonicalize_label(inj.label)
            res = rec.result or ScenarioResult(scenario_id=rec.spec.id)
            statuses = _aggregate_statuses(res.statuses)
            st = statuses.get(target, "not_triggered")
            lenient_det = _is_detected(st, strict=False)
            strict_det = _is_detected(st, strict=True)
            severity_map[sev].append((lenient_det, strict_det))

    severity_response: dict[str, SeverityResponseEntry] = {}
    for sev, outcomes in sorted(severity_map.items()):
        n_s = len(outcomes)
        l_rate = sum(o[0] for o in outcomes) / n_s if n_s > 0 else 0.0
        s_rate = sum(o[1] for o in outcomes) / n_s if n_s > 0 else 0.0
        severity_response[sev] = SeverityResponseEntry(
            severity=sev,
            n_scenarios=n_s,
            lenient_detection_rate=l_rate,
            strict_detection_rate=s_rate,
        )

    # 7. Timing Summary
    runtimes = [r.result.runtime_s for r in records if r.result is not None]
    if runtimes:
        timing = TimingSummary(
            mean_s=float(np.mean(runtimes)),
            median_s=float(np.median(runtimes)),
            min_s=float(np.min(runtimes)),
            max_s=float(np.max(runtimes)),
            total_s=float(np.sum(runtimes)),
        )
    else:
        timing = TimingSummary()

    # 8. Bootstrap Confidence Intervals
    bootstrap_ci: BootstrapCI | None = None
    if bootstrap and n_scenarios >= 5:
        rng = np.random.default_rng(seed)
        boot_macro_p: list[float] = []
        boot_macro_r: list[float] = []
        boot_macro_f1: list[float] = []
        boot_clean_fpr: list[float] = []

        for _ in range(n_boot):
            sample_idx = rng.integers(0, n_scenarios, size=n_scenarios)
            sample_records = [records[i] for i in sample_idx]

            # Fast scoring subset for macro lenient and clean control FPR
            _, b_macro = _evaluate_detection_sample(sample_records, all_targets, strict=False)
            if b_macro.precision is not None:
                boot_macro_p.append(b_macro.precision)
            if b_macro.recall is not None:
                boot_macro_r.append(b_macro.recall)
            if b_macro.f1 is not None:
                boot_macro_f1.append(b_macro.f1)

            # Clean control FPR sample
            cc_recs = [r for r in sample_records if r.spec.scenario_class == "clean_control"]
            if cc_recs:
                n_cc = len(cc_recs)
                n_fp = 0
                for r in cc_recs:
                    res = r.result or ScenarioResult(scenario_id=r.spec.id)
                    st_map = _aggregate_statuses(res.statuses)
                    if any(_is_detected(st, strict=False) for st in st_map.values()):
                        n_fp += 1
                boot_clean_fpr.append(n_fp / n_cc)

        def _ci(arr: list[float]) -> tuple[float, float] | None:
            if len(arr) < 10:
                return None
            return (float(np.percentile(arr, 2.5)), float(np.percentile(arr, 97.5)))

        bootstrap_ci = BootstrapCI(
            n_scenarios=n_scenarios,
            n_boot=n_boot,
            macro_precision=_ci(boot_macro_p),
            macro_recall=_ci(boot_macro_r),
            macro_f1=_ci(boot_macro_f1),
            clean_control_fpr=_ci(boot_clean_fpr),
        )

    return ScoreReport(
        n_scenarios=n_scenarios,
        lenient_metrics=lenient_metrics,
        strict_metrics=strict_metrics,
        macro_lenient=macro_lenient,
        macro_strict=macro_strict,
        fpr_by_class=fpr_by_class,
        attribution=attribution,
        reliability_table=reliability_table,
        severity_response=severity_response,
        timing=timing,
        bootstrap_ci=bootstrap_ci,
    )


def _evaluate_detection_sample(
    records: list[BenchmarkRecord],
    all_targets: set[str],
    strict: bool,
) -> tuple[dict[str, MetricScore], MetricScore]:
    """Helper used during bootstrap resampling."""
    per_target: dict[str, MetricScore] = {}

    for target in all_targets:
        tp = 0
        tp_recall = 0
        fp = 0
        fn = 0
        adj_fp = 0

        for rec in records:
            gt = rec.ground_truth
            res = rec.result or ScenarioResult(scenario_id=rec.spec.id)
            injected_labels = [_canonicalize_label(i.label) for i in gt.injected]
            is_injected = (target in injected_labels) or any(
                _type_level_name(i) == target for i in injected_labels
            )
            statuses = _aggregate_statuses(res.statuses)
            st = statuses.get(target, "not_triggered")
            detected = _is_detected(st, strict=strict)

            ctrl_statuses = _aggregate_statuses(res.control_statuses)
            ctrl_st = ctrl_statuses.get(target, "not_triggered")
            natural_in_control = _is_detected(ctrl_st, strict=strict)

            if is_injected:
                if gt.no_effect:
                    # Injected failure with negligible impact (<0.02)
                    # Excluded from recall; if detected, counts as FP (over-diagnosis)
                    if detected:
                        fp += 1
                        if not natural_in_control:
                            adj_fp += 1
                else:
                    if detected:
                        tp += 1
                        tp_recall += 1
                    else:
                        fn += 1
            else:
                if detected:
                    fp += 1
                    if not natural_in_control:
                        adj_fp += 1

        p, r, f, ap, af = _compute_p_r_f1(tp, tp_recall, fp, fn, adj_fp)
        per_target[target] = MetricScore(
            precision=p,
            recall=r,
            f1=f,
            tp=tp,
            fp=fp,
            fn=fn,
            adjusted_fp=adj_fp,
            adjusted_precision=ap,
            adjusted_f1=af,
        )

    p_vals = [m.precision for m in per_target.values() if m.precision is not None]
    r_vals = [m.recall for m in per_target.values() if m.recall is not None]
    f_vals = [m.f1 for m in per_target.values() if m.f1 is not None]
    ap_vals = [
        m.adjusted_precision for m in per_target.values() if m.adjusted_precision is not None
    ]
    af_vals = [m.adjusted_f1 for m in per_target.values() if m.adjusted_f1 is not None]

    macro = MetricScore(
        precision=float(np.mean(p_vals)) if p_vals else None,
        recall=float(np.mean(r_vals)) if r_vals else None,
        f1=float(np.mean(f_vals)) if f_vals else None,
        adjusted_precision=float(np.mean(ap_vals)) if ap_vals else None,
        adjusted_f1=float(np.mean(af_vals)) if af_vals else None,
        tp=sum(m.tp for m in per_target.values()),
        fp=sum(m.fp for m in per_target.values()),
        fn=sum(m.fn for m in per_target.values()),
        adjusted_fp=sum(m.adjusted_fp for m in per_target.values()),
    )
    return per_target, macro
