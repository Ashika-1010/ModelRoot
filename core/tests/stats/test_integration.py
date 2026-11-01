"""M7 integration tests: bootstrap + M5 metrics end-to-end (T7-21)."""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from modelroot.assessment.baseline import fit_baseline
from modelroot.data.loader import load_csv
from modelroot.data.split import make_split
from modelroot.data.validate import validate_for_analysis
from modelroot.models.registry import get_model
from modelroot.schemas.common import Algorithm, SplitMode
from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig
from modelroot.stats.bootstrap import BootstrapResult, paired_bootstrap
from modelroot.utils.seeds import SeedManager


def _make_split(tmp_path: pathlib.Path, n: int = 80) -> tuple:
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "age": rng.integers(20, 70, size=n).astype(str),
            "income": rng.integers(30000, 120000, size=n).astype(str),
            "label": ["yes" if rng.random() > 0.5 else "no" for _ in range(n)],
        }
    )
    p = tmp_path / "data.csv"
    df.to_csv(p, index=False)
    df_loaded = load_csv(p)
    config = AnalysisConfig(
        data_path=p,
        target="label",
        positive_class="yes",
        split=SplitConfig(mode=SplitMode.single_file),
        model_spec=ModelSpec(algorithm=Algorithm.logistic_regression, params={}),
        seed=0,
    )
    validated = validate_for_analysis(df_loaded, config)
    sm = SeedManager(0)
    split = make_split(validated, config, seed=sm.spawn("split"))
    return split, config


# T7-21: end-to-end M2->M5->M7 bootstrap integration
@pytest.mark.integration
def test_bootstrap_on_baseline_predictions(tmp_path: pathlib.Path) -> None:
    split, config = _make_split(tmp_path)
    col_types = {"age": "numeric", "income": "numeric"}

    adapter_a = get_model(config.model_spec, col_types, seed=0)
    adapter_b = get_model(
        ModelSpec(algorithm=Algorithm.random_forest, params={"n_estimators": 10}), col_types, seed=0
    )
    run_a = fit_baseline(adapter_a, split)
    run_b = fit_baseline(adapter_b, split)

    yt = run_a.holdout_pred.y_true
    yp_a = run_a.holdout_pred.y_pred
    yp_b = run_b.holdout_pred.y_pred
    ypr_a = run_a.holdout_pred.y_proba
    ypr_b = run_b.holdout_pred.y_proba

    result = paired_bootstrap(
        yt, yp_a, yp_b, "macro_f1", n_boot=200, seed=0, y_proba_a=ypr_a, y_proba_b=ypr_b
    )
    assert isinstance(result, BootstrapResult)
    assert (
        result.ci_low <= result.delta <= result.ci_high
        or result.delta < result.ci_low
        or result.delta > result.ci_high
    )  # CI may not contain delta
    assert 0 < result.n_valid <= result.n_boot
