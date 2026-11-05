"""Evidence and rule engine (M10).

Defines Criterion, Rule, RuleSet, RULE_SET_VERSION, the rule registry,
decision protocol, evidence strength, severity, ranking, and wording
validation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any

from modelroot.schemas.common import RulesProfile, Severity, Status

# Canonical rule-set version — single owner to avoid circular imports
RULE_SET_VERSION = "1.0"


# ---------------------------------------------------------------------------
# Criterion and Rule
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Criterion:
    """A single weighted criterion within a decision rule.

    Attributes:
        id: Stable identifier for this criterion.
        weight: Non-negative weight used in evidence-strength calculation.
        description: Human-readable description.
    """

    id: str
    weight: float = 1.0
    description: str = ""


@dataclass(frozen=True)
class Rule:
    """A versioned decision rule mapping evidence to a status.

    Attributes:
        id: Stable rule identifier (e.g. ``"OF-D1"``).
        text: Human-readable description of the rule.
        criteria: List of :class:`Criterion` objects.
    """

    id: str
    text: str
    criteria: list[Criterion] = field(default_factory=list)


# ---------------------------------------------------------------------------
# RuleSet and registry
# ---------------------------------------------------------------------------


class RuleSet:
    """A collection of decision rules keyed by rule ID.

    Attributes:
        version: Rule-set version string.
        rules: Mapping from rule_id to :class:`Rule`.
    """

    def __init__(self, version: str = RULE_SET_VERSION) -> None:
        self.version = version
        self._rules: dict[str, Rule] = {}

    def add(self, rule: Rule) -> None:
        """Register a rule.

        Args:
            rule: The :class:`Rule` to add.
        """
        self._rules[rule.id] = rule

    def get(self, rule_id: str) -> Rule | None:
        """Look up a rule by ID.

        Args:
            rule_id: Rule identifier.

        Returns:
            The :class:`Rule` or ``None`` if not found.
        """
        return self._rules.get(rule_id)

    @property
    def rules(self) -> dict[str, Rule]:
        """Return the internal rules mapping (read-only by convention)."""
        return self._rules

    def to_json(self) -> str:
        """Serialize the rule set to a JSON string.

        Returns:
            Deterministic JSON string.
        """
        data = {
            "version": self.version,
            "rules": {
                rid: {
                    "id": r.id,
                    "text": r.text,
                    "criteria": [
                        {"id": c.id, "weight": c.weight, "description": c.description}
                        for c in r.criteria
                    ],
                }
                for rid, r in sorted(self._rules.items())
            },
        }
        return json.dumps(data, indent=2, sort_keys=True)

    def to_markdown(self) -> str:
        """Serialize the rule set to Markdown.

        Returns:
            Human-readable Markdown string.
        """
        lines = [f"# Rule Set v{self.version}", ""]
        for rid in sorted(self._rules):
            r = self._rules[rid]
            lines.append(f"## {r.id}")
            lines.append(f"_{r.text}_")
            if r.criteria:
                lines.append("")
                lines.append("| Criterion | Weight | Description |")
                lines.append("|-----------|--------|-------------|")
                for c in r.criteria:
                    lines.append(f"| {c.id} | {c.weight} | {c.description} |")
            lines.append("")
        return "\n".join(lines)


# Global rule registry (in registration order)
_RULE_REGISTRY: list[Rule] = []


def register_rule(rule: Rule) -> Rule:
    """Register a rule in the global registry.

    Args:
        rule: The :class:`Rule` to register.

    Returns:
        *rule* unchanged.
    """
    if rule not in _RULE_REGISTRY:
        _RULE_REGISTRY.append(rule)
    return rule


def get_rule_registry() -> list[Rule]:
    """Return a shallow copy of the global rule registry."""
    return list(_RULE_REGISTRY)


def clear_rule_registry() -> None:
    """Clear the global rule registry (for testing only)."""
    _RULE_REGISTRY.clear()


# ---------------------------------------------------------------------------
# Decision protocol
# ---------------------------------------------------------------------------


def decide_intervention_effect(
    delta: float,
    ci_low: float,
    ci_high: float,
    gate_ok: bool,
    extra_criteria: dict[str, bool],
    thresholds: Any,  # Thresholds
    profile: RulesProfile,
) -> tuple[Status, dict[str, bool]]:
    """Apply the generic decision protocol from §5.

    Args:
        delta: Point estimate of the effect.
        ci_low: Lower CI bound.
        ci_high: Upper CI bound.
        gate_ok: Whether the power gate passed.
        extra_criteria: Additional named criteria that must all be True
            for ``supported``.
        thresholds: :class:`~modelroot.config.thresholds.Thresholds` instance.
        profile: ``full`` or ``naive``.

    Returns:
        Tuple of (Status, criteria_met dict).
    """
    delta_min = thresholds.delta_min
    criteria_met: dict[str, bool] = {}

    if profile == RulesProfile.naive:
        criteria_met["delta_gte_delta_min"] = delta >= delta_min
        status = Status.supported if delta >= delta_min else Status.rejected
        return status, criteria_met

    # Full protocol
    criteria_met["gate_ok"] = gate_ok
    if not gate_ok:
        criteria_met["gate_ok"] = False
        return Status.inconclusive, criteria_met

    # Equivalence-style rejection
    criteria_met["ci_high_lt_delta_min"] = ci_high < delta_min
    if ci_high < delta_min:
        return Status.rejected, criteria_met

    # Supported: delta >= delta_min, ci_low > 0, ALL extra criteria
    criteria_met["delta_gte_delta_min"] = delta >= delta_min
    criteria_met["ci_low_gt_0"] = ci_low > 0
    all_extra = all(extra_criteria.values()) if extra_criteria else True
    for k, v in extra_criteria.items():
        criteria_met[k] = v

    if delta >= delta_min and ci_low > 0 and all_extra:
        return Status.supported, criteria_met

    # Weakly supported: delta >= delta_min/2 and delta > 0
    criteria_met["delta_gte_delta_min_half"] = delta >= delta_min / 2
    if delta >= delta_min / 2 and delta > 0:
        return Status.weakly_supported, criteria_met

    # Otherwise inconclusive
    return Status.inconclusive, criteria_met


# ---------------------------------------------------------------------------
# Evidence strength
# ---------------------------------------------------------------------------


def evidence_strength(rule: Rule, criteria_met: dict[str, bool]) -> float:
    """Compute the weighted fraction of satisfied criteria.

    strength = sum(weights of met criteria) / sum(all weights)

    Args:
        rule: The decision :class:`Rule`.
        criteria_met: Map from criterion ID to whether it was met.

    Returns:
        Float in [0, 1].
    """
    total_weight = sum(c.weight for c in rule.criteria)
    if total_weight <= 0:
        return 0.0
    met_weight = sum(c.weight for c in rule.criteria if criteria_met.get(c.id, False))
    return float(met_weight / total_weight)


# ---------------------------------------------------------------------------
# Severity
# ---------------------------------------------------------------------------


def severity(delta: float, thresholds: Any) -> Severity | None:
    """Classify the severity of an effect.

    Δ >= 0.05 → high, 0.02 <= Δ < 0.05 → medium, Δ < 0.02 → low.

    Args:
        delta: Absolute effect size (|Δ|).
        thresholds: :class:`Thresholds` instance.

    Returns:
        :class:`Severity` or ``None`` if delta is not finite.
    """
    if not math.isfinite(delta):
        return None
    abs_delta = abs(delta)
    if abs_delta >= 0.05:
        return Severity.high
    if abs_delta >= 0.02:
        return Severity.medium
    return Severity.low


# ---------------------------------------------------------------------------
# Ranking
# ---------------------------------------------------------------------------


_STATUS_ORDER: dict[Status, int] = {
    Status.supported: 0,
    Status.weakly_supported: 1,
    Status.inconclusive: 2,
    Status.rejected: 3,
    Status.not_triggered: 4,
    Status.not_applicable: 5,
}


def rank(results: list[Any]) -> list[Any]:
    """Rank hypothesis results by (status order, effect delta desc, evidence_strength desc, id).

    Args:
        results: List of :class:`HypothesisResult` objects.

    Returns:
        The sorted list (same objects, new order).
    """

    def _sort_key(r: Any) -> tuple[int, float, float, str]:
        status_rank = _STATUS_ORDER.get(r.status, 99)
        effect_delta = r.effect.delta if r.effect is not None else 0.0
        strength = r.evidence_strength if hasattr(r, "evidence_strength") else 0.0
        hyp_id = r.id if hasattr(r, "id") else ""
        return (status_rank, -effect_delta, -strength, hyp_id)

    return sorted(results, key=_sort_key)


# ---------------------------------------------------------------------------
# Wording validation
# ---------------------------------------------------------------------------


_BANNED_PHRASES: list[str] = [
    "proves",
    "proven",
    "confirmed",
    "definitely",
    "root cause is",
    "caused by",
]


def check_wording(text: str, status: Status) -> list[str]:
    """Check that explanation text is consistent with the assigned status.

    Args:
        text: The explanation text.
        status: The verdict status.

    Returns:
        List of violation messages (empty if clean).
    """
    violations: list[str] = []
    lower = text.lower()
    for phrase in _BANNED_PHRASES:
        if phrase in lower:
            violations.append(f"Banned phrase '{phrase}' found in explanation.")
    return violations


# ---------------------------------------------------------------------------
# Evidence builders
# ---------------------------------------------------------------------------


def make_evidence_id(name: str, value: float, method: str, experiment_id: str | None) -> str:
    """Compute a stable evidence ID.

    The ID is ``"ev_"`` followed by 8 hex characters derived from a SHA-256
    hash of ``(name, value, method, experiment_id)``.

    Args:
        name: Evidence name.
        value: Evidence value.
        method: Statistical method.
        experiment_id: Optional experiment ID.

    Returns:
        ``"ev_"`` + 8 hex chars.
    """
    import hashlib

    payload = f"{name}|{value}|{method}|{experiment_id}"
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:8]
    return f"ev_{digest}"


# ---------------------------------------------------------------------------
# Finalize — integration into diagnose()
# ---------------------------------------------------------------------------


def finalize(
    hyp_results: list[Any],
    thresholds: Any,
) -> list[Any]:
    """Apply M10 finalization to hypothesis results before returning AnalysisResult.

    Steps:
      1. Preserve hypothesis-provided evidence_strength (no overwrite).
      2. Assign severity to supported/weakly_supported results that have an
         effect and no existing severity.
      3. Rank all results by (status order, delta desc, strength desc, id).
      4. Validate unique hypothesis IDs.
      5. Reject non-finite floats before constructing the final result.

    Args:
        hyp_results: List of :class:`HypothesisResult` objects.
        thresholds: :class:`Thresholds` instance.

    Returns:
        The finalized and ranked list of :class:`HypothesisResult` objects.

    Raises:
        ValueError: If duplicate IDs are found or non-finite values are present.
    """
    import math

    # 4. Validate unique IDs
    ids = [r.id for r in hyp_results]
    if len(ids) != len(set(ids)):
        dupes = [i for i in ids if ids.count(i) > 1]
        raise ValueError(f"Duplicate hypothesis result IDs: {sorted(set(dupes))}")

    # 5. Reject non-finite floats
    for r in hyp_results:
        if not math.isfinite(r.evidence_strength):
            raise ValueError(f"Non-finite evidence_strength in {r.id!r}")
        if r.effect is not None:
            for field_name in ("delta", "ci_low", "ci_high", "baseline", "intervention"):
                val = getattr(r.effect, field_name)
                if not math.isfinite(val):
                    raise ValueError(f"Non-finite {field_name} in effect for {r.id!r}")

    # 2. Assign missing severity
    for r in hyp_results:
        if (
            r.severity is None
            and r.status in (Status.supported, Status.weakly_supported)
            and r.effect is not None
        ):
            sev = severity(r.effect.delta, thresholds)
            if sev is not None:
                # Use model_copy to update the frozen-ish result
                r = r.model_copy(update={"severity": sev})
                # Replace in list — find by index
                idx = hyp_results.index(next(h for h in hyp_results if h.id == r.id))
                hyp_results[idx] = r

    # 3. Rank
    ranked = rank(hyp_results)

    return ranked
