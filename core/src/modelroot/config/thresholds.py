"""Assessment thresholds (M6).

``Thresholds`` is the single source of truth for all numerical decision
boundaries used by the assessment verdict logic.  Only assessor-level
fields are included in M6; hypothesis thresholds are added in later
milestones.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Thresholds(BaseModel):
    """Numerical thresholds for the assessment verdict (M6 scope).

    All fields are frozen after construction.  Future milestones will add
    hypothesis-specific thresholds via additive extension.

    Attributes:
        headroom: Minimum gap (reference - baseline) on the primary metric
            that triggers the "headroom" underperformance reason.
            Default 0.03 per §4.3.
        gap: Minimum train-holdout gap on the primary metric that triggers
            the "gap" underperformance reason.  Default 0.10 per §4.3.
        current_drop: Minimum holdout-current drop on the primary metric
            that triggers the "current drop" underperformance reason.
            Default 0.03 per §4.3.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    headroom: float = 0.03
    gap: float = 0.10
    current_drop: float = 0.03
