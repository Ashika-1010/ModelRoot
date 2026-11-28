"""Tests for M20 benchmark injector (shift)."""

import numpy as np
import pandas as pd

from modelroot_bench.datasets.registry import DatasetSpec
from modelroot_bench.inject.shift import ShiftInjector


def _make_dummy_spec() -> DatasetSpec:
    return DatasetSpec(
        name="test_data",
        load=lambda: pd.DataFrame({"x": [1, 2], "target": [0, 1]}),
        target="target",
        positive="1",
        role="dev",
        description="test",
    )


def test_shift_injector() -> None:
    n = 1500
    rng = np.random.default_rng(42)
    df = pd.DataFrame(
        {
            "f1": rng.normal(size=n),
            "f2": rng.normal(size=n),
            "target": rng.integers(0, 2, size=n),
        }
    )

    inj = ShiftInjector()
    spec = _make_dummy_spec()
    scen = inj.apply(df, spec, severity="S2", seed=42)

    assert scen.current_df is not None
    assert len(scen.current_df) > 0
    assert scen.ground_truth.injected[0].label == "shift"
    assert scen.direction == "degradation"
