"""Assessment thresholds (M6 + M8).

``Thresholds`` is the single source of truth for all numerical decision
boundaries used by the assessment verdict logic and the M8 power gate.
Only assessor-level and M8 fields are included here; hypothesis thresholds
are added in later milestones.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Thresholds(BaseModel):
    """Numerical thresholds for assessment verdict and statistics (M6/M8 scope).

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
        delta_min: Practical-significance threshold used by the power gate
            and effect classification.  Default 0.02 per §4.1.
        alpha: Significance level for bootstrap CIs.  Default 0.05 per §4.1.
        n_boot: Default number of bootstrap resamples.  Default 1000 per §4.1.
        repeats: Default number of repeated subsampling repeats.
            Default 10 per §4.1.
        holdout_min_n: Minimum holdout set size for the power gate.
            Default 200 per §4.2.
        holdout_min_pos: Minimum positive-class holdout samples for the
            power gate.  Default 30 per §4.2.
        ci_width_max: Maximum bootstrap CI width before the power gate
            flags underpowered.  Default 0.10 per §4.2.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    # M6 assessor thresholds
    headroom: float = 0.03
    gap: float = 0.10
    current_drop: float = 0.03

    # M8 statistics thresholds
    delta_min: float = 0.02
    alpha: float = 0.05
    n_boot: int = 1000
    repeats: int = 10
    holdout_min_n: int = 200
    holdout_min_pos: int = 30
    ci_width_max: float = 0.10
