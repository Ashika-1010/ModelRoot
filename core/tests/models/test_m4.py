"""M4 tests: Model adapters and registry.

Test types:
  U   = unit
  D   = deterministic fixture
  P   = property-based (Hypothesis)
  Sec = security
  I   = integration (M2→M3→M4 pipeline)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from modelroot.data.loader import load_csv
from modelroot.data.split import make_split
from modelroot.data.validate import validate_for_analysis
from modelroot.models.adapter import ModelAdapter
from modelroot.models.registry import InvalidModelSpec, get_model, list_model_specs
from modelroot.schemas.common import Algorithm, SplitMode
from modelroot.schemas.config import AnalysisConfig, ModelSpec, SplitConfig

_FIXTURES = pathlib.Path(__file__).parent.parent / "fixtures" / "csv"


# ---------------------------------------------------------------------------
# Tiny synthetic dataset fixture
# ---------------------------------------------------------------------------


def _tiny_split(
    n: int = 40,
    n_cat: bool = True,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series, dict[str, str]]:
    """Return (X_train, y_train, X_holdout, y_holdout, column_types)."""
    import tempfile

    with tempfile.TemporaryDirectory() as td:
        rows = "\n".join(
            f"{20 + i},{50000 + i * 1000},"
            f"{'M' if i % 2 == 0 else 'F'},{'yes' if i % 2 == 0 else 'no'}"
            for i in range(n)
        )
        p = pathlib.Path(td) / "data.csv"
        p.write_text("age,income,gender,label\n" + rows, encoding="utf-8")
        df = load_csv(p)
        cfg = AnalysisConfig(
            data_path=p,
            target="label",
            positive_class="yes",
            split=SplitConfig(mode=SplitMode.single_file),
            model_spec=ModelSpec(algorithm=Algorithm.logistic_regression),
        )
        vd = validate_for_analysis(df, cfg)
        split = make_split(vd, cfg, 0)
        return (
            split.X_train,
            split.y_train,
            split.X_holdout,
            split.y_holdout,
            vd.column_types,
        )


# Cache at module level (expensive to rebuild each test on Windows)
_X_TRAIN, _Y_TRAIN, _X_HOLDOUT, _Y_HOLDOUT, _COL_TYPES = _tiny_split()


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _make_adapter(algo: Algorithm, extra_params: dict | None = None) -> ModelAdapter:
    spec = ModelSpec(algorithm=algo, params=extra_params or {})
    return get_model(spec, _COL_TYPES, seed=0)


# ---------------------------------------------------------------------------
# U: get_model & list_model_specs
# ---------------------------------------------------------------------------


def test_get_model_returns_model_adapter() -> None:
    """U: get_model returns a ModelAdapter."""
    adapter = _make_adapter(Algorithm.logistic_regression)
    assert isinstance(adapter, ModelAdapter)


def test_list_model_specs_has_all_algorithms() -> None:
    """U: list_model_specs returns one entry per algorithm."""
    specs = list_model_specs()
    algo_names = {s["algorithm"] for s in specs}
    expected = {a.value for a in Algorithm}
    assert algo_names == expected


def test_invalid_algorithm_raises() -> None:
    """U: Unknown algorithm string raises a validation error."""
    with pytest.raises(Exception):  # noqa: B017
        ModelSpec(algorithm="xgboost", params={})  # type: ignore[arg-type]


def test_invalid_param_raises_invalid_model_spec() -> None:
    """U: Unknown parameter raises InvalidModelSpec."""
    with pytest.raises(InvalidModelSpec, match="Unknown parameter"):
        _make_adapter(Algorithm.logistic_regression, {"unknown_param": 1})


def test_security_injection_string_rejected() -> None:
    """Sec: String param with injection-like content is rejected."""
    with pytest.raises(InvalidModelSpec):
        _make_adapter(
            Algorithm.logistic_regression,
            {"penalty": "__import__('os').system('id')"},
        )


def test_security_C_out_of_range_rejected() -> None:
    """Sec/U: C=0 (out of range) is rejected."""
    with pytest.raises(InvalidModelSpec, match=">="):
        _make_adapter(Algorithm.logistic_regression, {"C": 0.0})


# ---------------------------------------------------------------------------
# U: fit / predict / predict_proba — all four adapters
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.logistic_regression,
        Algorithm.decision_tree,
        Algorithm.random_forest,
        Algorithm.hist_gradient_boosting,
    ],
)
def test_adapter_fit_predict(algo: Algorithm) -> None:
    """U: Each adapter fits and predicts without error."""
    adapter = _make_adapter(algo)
    adapter.fit(_X_TRAIN, _Y_TRAIN)
    preds = adapter.predict(_X_HOLDOUT)
    assert preds.shape == (len(_Y_HOLDOUT),)
    assert set(preds).issubset({0, 1})


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.logistic_regression,
        Algorithm.decision_tree,
        Algorithm.random_forest,
        Algorithm.hist_gradient_boosting,
    ],
)
def test_adapter_predict_proba_shape(algo: Algorithm) -> None:
    """U: predict_proba returns (n, 2) array."""
    adapter = _make_adapter(algo)
    adapter.fit(_X_TRAIN, _Y_TRAIN)
    proba = adapter.predict_proba(_X_HOLDOUT)
    assert proba.shape == (len(_Y_HOLDOUT), 2)


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.logistic_regression,
        Algorithm.decision_tree,
        Algorithm.random_forest,
        Algorithm.hist_gradient_boosting,
    ],
)
def test_adapter_proba_sums_to_one(algo: Algorithm) -> None:
    """U: Each row of predict_proba sums to 1.0 ± 1e-9."""
    adapter = _make_adapter(algo)
    adapter.fit(_X_TRAIN, _Y_TRAIN)
    proba = adapter.predict_proba(_X_HOLDOUT)
    row_sums = proba.sum(axis=1)
    np.testing.assert_allclose(row_sums, 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# D: determinism — same seed → same predictions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.logistic_regression,
        Algorithm.decision_tree,
        Algorithm.random_forest,
        Algorithm.hist_gradient_boosting,
    ],
)
def test_same_seed_identical_predictions(algo: Algorithm) -> None:
    """D: Two adapters with the same seed produce identical predictions."""
    spec = ModelSpec(algorithm=algo)
    a1 = get_model(spec, _COL_TYPES, seed=42)
    a2 = get_model(spec, _COL_TYPES, seed=42)
    a1.fit(_X_TRAIN, _Y_TRAIN)
    a2.fit(_X_TRAIN, _Y_TRAIN)
    np.testing.assert_array_equal(a1.predict(_X_HOLDOUT), a2.predict(_X_HOLDOUT))
    np.testing.assert_allclose(
        a1.predict_proba(_X_HOLDOUT), a2.predict_proba(_X_HOLDOUT), atol=1e-12
    )


# ---------------------------------------------------------------------------
# U: clone_with — immutability and param override
# ---------------------------------------------------------------------------


def test_clone_with_does_not_mutate_original() -> None:
    """U: clone_with leaves the original adapter unchanged."""
    original = _make_adapter(Algorithm.logistic_regression, {"C": 1.0})
    cloned = original.clone_with(C=0.01)
    assert original.params["C"] == 1.0
    assert cloned.params["C"] == 0.01


def test_clone_with_invalid_param_raises() -> None:
    """U: clone_with with invalid param raises InvalidModelSpec."""
    adapter = _make_adapter(Algorithm.logistic_regression)
    with pytest.raises(InvalidModelSpec, match="Unknown parameter"):
        adapter.clone_with(bad_param=99)


# ---------------------------------------------------------------------------
# U: capacity_grid
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.logistic_regression,
        Algorithm.decision_tree,
        Algorithm.random_forest,
        Algorithm.hist_gradient_boosting,
    ],
)
def test_capacity_grid_all_entries_fit(algo: Algorithm) -> None:
    """U: Every capacity-grid entry produces an adapter that fits on tiny data."""
    base = _make_adapter(algo)
    grid = base.capacity_grid()
    assert len(grid) > 0
    for params in grid:
        cloned = base.clone_with(**params)
        cloned.fit(_X_TRAIN, _Y_TRAIN)
        preds = cloned.predict(_X_HOLDOUT)
        assert len(preds) == len(_Y_HOLDOUT)


def test_lr_capacity_grid_exact() -> None:
    """U: LR capacity grid has exactly 4 C values as per §5-H1."""
    grid = _make_adapter(Algorithm.logistic_regression).capacity_grid()
    c_values = sorted([g["C"] for g in grid])
    assert c_values == sorted([1.0, 0.1, 0.01, 0.001])


def test_dt_capacity_grid_size() -> None:
    """U: DT capacity grid has 3 x 2 = 6 entries."""
    grid = _make_adapter(Algorithm.decision_tree).capacity_grid()
    assert len(grid) == 6


def test_rf_capacity_grid_size() -> None:
    """U: RF capacity grid has 2 x 2 x 2 = 8 entries."""
    grid = _make_adapter(Algorithm.random_forest).capacity_grid()
    assert len(grid) == 8


# ---------------------------------------------------------------------------
# U: supports_sample_weight
# ---------------------------------------------------------------------------


def test_lr_does_not_support_sample_weight() -> None:
    """U: LogisticRegression does not support sample_weight."""
    assert not _make_adapter(Algorithm.logistic_regression).supports_sample_weight()


def test_dt_supports_sample_weight() -> None:
    """U: DecisionTree supports sample_weight."""
    assert _make_adapter(Algorithm.decision_tree).supports_sample_weight()


def test_rf_supports_sample_weight() -> None:
    """U: RandomForest supports sample_weight."""
    assert _make_adapter(Algorithm.random_forest).supports_sample_weight()


def test_hgb_supports_sample_weight() -> None:
    """U: HistGradientBoosting supports sample_weight."""
    assert _make_adapter(Algorithm.hist_gradient_boosting).supports_sample_weight()


def test_sample_weight_accepted_by_dt(tmp_path: pathlib.Path) -> None:
    """U: DecisionTree accepts sample_weight without error."""
    adapter = _make_adapter(Algorithm.decision_tree)
    sw = np.ones(len(_Y_TRAIN))
    sw[0] = 2.0  # give first sample double weight
    adapter.fit(_X_TRAIN, _Y_TRAIN, sample_weight=sw)
    preds = adapter.predict(_X_HOLDOUT)
    assert len(preds) == len(_Y_HOLDOUT)


# ---------------------------------------------------------------------------
# U: complexity()
# ---------------------------------------------------------------------------


def test_lr_complexity_increases_with_C() -> None:
    """U: LR complexity increases with C (less regularization = more complex)."""
    c1 = get_model(
        ModelSpec(algorithm=Algorithm.logistic_regression, params={"C": 0.01}), _COL_TYPES, 0
    )
    c2 = get_model(
        ModelSpec(algorithm=Algorithm.logistic_regression, params={"C": 10.0}), _COL_TYPES, 0
    )
    assert c1.complexity() < c2.complexity()


# ---------------------------------------------------------------------------
# U: param validation boundary cases
# ---------------------------------------------------------------------------


def test_rf_n_estimators_over_500_rejected() -> None:
    """U: n_estimators > 500 is rejected for RF."""
    with pytest.raises(InvalidModelSpec, match="<= 500"):
        _make_adapter(Algorithm.random_forest, {"n_estimators": 501})


def test_hgb_max_iter_over_500_rejected() -> None:
    """U: max_iter > 500 is rejected for HGB."""
    with pytest.raises(InvalidModelSpec, match="<= 500"):
        _make_adapter(Algorithm.hist_gradient_boosting, {"max_iter": 501})


def test_dt_class_weight_balanced_accepted() -> None:
    """U: class_weight='balanced' is valid for DT."""
    adapter = _make_adapter(Algorithm.decision_tree, {"class_weight": "balanced"})
    assert adapter.params["class_weight"] == "balanced"


# ---------------------------------------------------------------------------
# P: property-based — clone_with round-trips
# ---------------------------------------------------------------------------


@given(c_val=st.floats(min_value=1e-4, max_value=100.0, allow_nan=False))
@settings(max_examples=20)
def test_lr_clone_with_c_roundtrip(c_val: float) -> None:
    """P: Any valid C value round-trips through clone_with."""
    base = _make_adapter(Algorithm.logistic_regression)
    cloned = base.clone_with(C=c_val)
    assert abs(cloned.params["C"] - c_val) < 1e-10


# ---------------------------------------------------------------------------
# I: Full M2→M3→M4 pipeline integration
# ---------------------------------------------------------------------------


def test_full_pipeline_lr(tmp_path: pathlib.Path) -> None:
    """I: Full pipeline: load → validate → split → LR fit/predict."""
    rows = "\n".join(
        f"{20 + i},{50000 + i * 1000},{'yes' if i % 2 == 0 else 'no'}" for i in range(40)
    )
    p = tmp_path / "data.csv"
    p.write_text("age,income,label\n" + rows, encoding="utf-8")
    df = load_csv(p)
    cfg = AnalysisConfig(
        data_path=p,
        target="label",
        positive_class="yes",
        split=SplitConfig(mode=SplitMode.single_file),
        model_spec=ModelSpec(algorithm=Algorithm.logistic_regression),
    )
    vd = validate_for_analysis(df, cfg)
    from modelroot.data.split import make_split

    split = make_split(vd, cfg, 42)
    adapter = get_model(cfg.model_spec, vd.column_types, seed=42)
    adapter.fit(split.X_train, split.y_train)
    proba = adapter.predict_proba(split.X_holdout)
    assert proba.shape[1] == 2
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


@pytest.mark.parametrize(
    "algo",
    [
        Algorithm.logistic_regression,
        Algorithm.decision_tree,
        Algorithm.random_forest,
        Algorithm.hist_gradient_boosting,
    ],
)
def test_all_algorithms_full_pipeline(algo: Algorithm, tmp_path: pathlib.Path) -> None:
    """I: All four algorithms run end-to-end through M2→M3→M4."""
    rows = "\n".join(
        f"{20 + i},{50000 + i * 1000},{'M' if i % 3 == 0 else 'F'},{'yes' if i % 2 == 0 else 'no'}"
        for i in range(40)
    )
    p = tmp_path / f"data_{algo.value}.csv"
    p.write_text("age,income,gender,label\n" + rows, encoding="utf-8")
    df = load_csv(p)
    cfg = AnalysisConfig(
        data_path=p,
        target="label",
        positive_class="yes",
        split=SplitConfig(mode=SplitMode.single_file),
        model_spec=ModelSpec(algorithm=algo),
    )
    vd = validate_for_analysis(df, cfg)
    from modelroot.data.split import make_split

    split = make_split(vd, cfg, 0)
    adapter = get_model(cfg.model_spec, vd.column_types, seed=0)
    adapter.fit(split.X_train, split.y_train)
    proba = adapter.predict_proba(split.X_holdout)
    assert proba.shape == (len(split.y_holdout), 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-9)


# ---------------------------------------------------------------------------
# U: LR penalty parameter — correct forwarding to sklearn estimator
# ---------------------------------------------------------------------------


def test_lr_penalty_l2_uses_C_and_l1_ratio_zero() -> None:
    """U: penalty='l2' produces an estimator with finite C and l1_ratio=0.0."""
    adapter = _make_adapter(Algorithm.logistic_regression, {"penalty": "l2", "C": 0.5})
    adapter.fit(_X_TRAIN, _Y_TRAIN)
    estimator = adapter._pipeline.named_steps["estimator"]  # type: ignore[union-attr]
    assert estimator.C == 0.5
    assert estimator.l1_ratio == 0.0


def test_lr_penalty_none_string_uses_infinite_C() -> None:
    """U: penalty='none' produces an estimator with C=inf (no regularization)."""
    adapter = _make_adapter(Algorithm.logistic_regression, {"penalty": "none"})
    adapter.fit(_X_TRAIN, _Y_TRAIN)
    estimator = adapter._pipeline.named_steps["estimator"]  # type: ignore[union-attr]
    assert float("inf") == estimator.C
    assert estimator.l1_ratio == 0.0


def test_lr_penalty_none_value_rejected() -> None:
    """U: penalty=None is rejected (master plan specifies {l2, none} as strings only)."""
    with pytest.raises(InvalidModelSpec, match="penalty"):
        _make_adapter(Algorithm.logistic_regression, {"penalty": None})


def test_lr_penalty_l2_vs_none_produces_different_coefficients() -> None:
    """U: penalty='l2' with small C shrinks coefficients vs penalty='none'."""
    reg = _make_adapter(Algorithm.logistic_regression, {"penalty": "l2", "C": 0.001})
    unreg = _make_adapter(Algorithm.logistic_regression, {"penalty": "none"})
    reg.fit(_X_TRAIN, _Y_TRAIN)
    unreg.fit(_X_TRAIN, _Y_TRAIN)
    reg_coef = reg._pipeline.named_steps["estimator"].coef_  # type: ignore[union-attr]
    unreg_coef = unreg._pipeline.named_steps["estimator"].coef_  # type: ignore[union-attr]
    # Strong L2 regularization (C=0.001) should shrink coefficients
    assert np.abs(reg_coef).sum() < np.abs(unreg_coef).sum()


def test_lr_fit_no_future_warning_for_penalty() -> None:
    """U: Fitting with penalty='l2' does not emit a FutureWarning about 'penalty'."""
    import warnings

    adapter = _make_adapter(Algorithm.logistic_regression, {"penalty": "l2"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.fit(_X_TRAIN, _Y_TRAIN)
    penalty_warnings = [
        w
        for w in caught
        if "penalty" in str(w.message).lower() and "deprecated" in str(w.message).lower()
    ]
    assert len(penalty_warnings) == 0, (
        "Expected no FutureWarning about 'penalty', but got: "
        f"{[str(w.message) for w in penalty_warnings]}"
    )


def test_lr_fit_no_future_warning_for_penalty_none() -> None:
    """U: Fitting with penalty='none' does not emit a FutureWarning about 'penalty'."""
    import warnings

    adapter = _make_adapter(Algorithm.logistic_regression, {"penalty": "none"})
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        adapter.fit(_X_TRAIN, _Y_TRAIN)
    penalty_warnings = [
        w
        for w in caught
        if "penalty" in str(w.message).lower() and "deprecated" in str(w.message).lower()
    ]
    assert len(penalty_warnings) == 0, (
        "Expected no FutureWarning about 'penalty', but got: "
        f"{[str(w.message) for w in penalty_warnings]}"
    )


def test_lr_penalty_l1_rejected() -> None:
    """U: penalty='l1' is rejected (not in allowed values {l2, none, None})."""
    with pytest.raises(InvalidModelSpec, match="penalty"):
        _make_adapter(Algorithm.logistic_regression, {"penalty": "l1"})
