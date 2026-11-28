"""Assessment thresholds (M6 + M8 + M11 + M14 + M15 + M16).

``Thresholds`` is the single source of truth for all numerical decision
boundaries used by the assessment verdict logic, M8 power gate, and
hypothesis evaluations.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class Thresholds(BaseModel):
    """Numerical thresholds for assessment verdict and statistics (M6-M16 scope).

    All fields are frozen after construction.

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

    # M11 overfitting hypothesis thresholds
    gap_rel: float = 0.15
    gap_shrink_min: float = 0.30
    consistency_min: float = 0.80

    # M14 class imbalance hypothesis thresholds
    minority_share: float = 0.20
    acc_bacc_gap: float = 0.05
    majority_margin: float = 0.02
    minority_oof_min: int = 30

    # M15 structural leakage thresholds
    dup_frac: float = 0.01
    id_unique_ratio: float = 0.95
    id_rowcorr: float = 0.95
    id_imp_supported: float = 0.25
    id_imp_rejected: float = 0.05

    # M16 target-proxy leakage thresholds
    sf_auc: float = 0.90
    proxy_imp_share: float = 0.50
    transfer_auc_drop: float = 0.15
    flagged_imp_supported: float = 0.10
    flagged_imp_min: float = 0.05

    # M18 subgroup underperformance thresholds
    slice_gap: float = 0.05
    slice_min_n: int = 30
    slice_fdr_q: float = 0.10
    slice_max_candidates: int = 5

    # M19/M20 distribution shift thresholds
    drift_auc_min: float = 0.60
    drift_p_max: float = 0.01
    drift_ks_min: float = 0.10
    drift_psi_min: float = 0.20
    dose_rho_min: float = 0.0
    dose_p_max: float = 0.05
    dose_min_rows: int = 50
    dose_min_bins: int = 5
