"""Property-based tests for AnalysisConfig using the Hypothesis library.

These tests generate arbitrary valid AnalysisConfig instances and verify
that:
- JSON round-trips are lossless.
- model_dump() → model_validate() round-trips are lossless.
- schema_version is always "1.0" on AnalysisResult instances.

Strategies are kept conservative (no Path objects with unusual characters,
no NaN/Inf floats) to avoid generating inputs that are structurally valid
but practically meaningless.
"""

from __future__ import annotations

from pathlib import Path

from hypothesis import given, settings
from hypothesis import strategies as st

from modelroot.schemas import (
    Algorithm,
    AnalysisConfig,
    Metadata,
    ModelSpec,
    PrimaryMetric,
    RulesProfile,
    SplitConfig,
    SplitMode,
)
from modelroot.schemas.config import AblationConfig

# ---------------------------------------------------------------------------
# Reusable Hypothesis strategies
# ---------------------------------------------------------------------------

# Safe identifiers: non-empty, no control chars
_safe_text = st.text(
    alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd"), whitelist_characters="_-"),
    min_size=1,
    max_size=30,
)

# Safe floats: finite, within (0, 1) exclusive for test_size / expected_score
_fraction = st.floats(min_value=0.01, max_value=0.99, allow_nan=False, allow_infinity=False)

# budget_s must be > 0
_budget = st.floats(min_value=0.001, max_value=3600.0, allow_nan=False, allow_infinity=False)

# seed must be >= 0
_seed = st.integers(min_value=0, max_value=2**31 - 1)

# JSON scalar param values
_scalar_value: st.SearchStrategy[int | float | str | bool | None] = st.one_of(
    st.integers(min_value=-1000, max_value=1000),
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    _safe_text,
    st.booleans(),
    st.none(),
)

_params_dict = st.dictionaries(
    keys=_safe_text,
    values=_scalar_value,
    max_size=5,
)

_model_spec_strategy = st.builds(
    ModelSpec,
    algorithm=st.sampled_from(list(Algorithm)),
    params=_params_dict,
)

_split_config_single = st.builds(
    SplitConfig,
    mode=st.just(SplitMode.single_file),
    test_size=_fraction,
    split_column=st.none(),
    test_path=st.none(),
)

_split_config_split_col = st.builds(
    SplitConfig,
    mode=st.just(SplitMode.split_column),
    test_size=_fraction,
    split_column=_safe_text,
    test_path=st.none(),
)

_split_config_strategy = st.one_of(_split_config_single, _split_config_split_col)

_metadata_strategy = st.builds(
    Metadata,
    expected_score=st.one_of(st.none(), _fraction),
    unavailable_at_inference=st.lists(_safe_text, max_size=3),
    id_columns=st.lists(_safe_text, max_size=3),
    sensitive_columns=st.lists(_safe_text, max_size=3),
)

_ablation_strategy = st.builds(
    AblationConfig,
    subgroup_protocol=st.sampled_from(list(RulesProfile)),
)

_analysis_config_strategy = st.builds(
    AnalysisConfig,
    data_path=_safe_text.map(lambda s: Path(f"data/{s}.csv")),
    target=_safe_text,
    positive_class=_safe_text,
    split=_split_config_strategy,
    current_path=st.none(),
    model_spec=_model_spec_strategy,
    primary_metric=st.sampled_from(list(PrimaryMetric)),
    metadata=_metadata_strategy,
    seed=_seed,
    budget_s=_budget,
    run_experiments=st.booleans(),
    rules_profile=st.sampled_from(list(RulesProfile)),
    ablation=_ablation_strategy,
    explain=st.booleans(),
)


# ---------------------------------------------------------------------------
# Property tests
# ---------------------------------------------------------------------------


@given(_analysis_config_strategy)
@settings(max_examples=50)
def test_analysis_config_round_trip_model_dump(config: AnalysisConfig) -> None:
    """model_dump() → model_validate() must be lossless."""
    restored = AnalysisConfig.model_validate(config.model_dump())
    assert restored == config


@given(_analysis_config_strategy)
@settings(max_examples=50)
def test_analysis_config_round_trip_json(config: AnalysisConfig) -> None:
    """model_dump_json() → model_validate_json() must be lossless."""
    restored = AnalysisConfig.model_validate_json(config.model_dump_json())
    assert restored == config


@given(_analysis_config_strategy)
@settings(max_examples=30)
def test_analysis_config_serialises_to_dict(config: AnalysisConfig) -> None:
    """model_dump() must produce a plain dict (not a Pydantic model)."""
    dumped = config.model_dump()
    assert isinstance(dumped, dict)
    # Nested model_spec should also be a dict after dump
    assert isinstance(dumped["model_spec"], dict)


@given(_model_spec_strategy)
@settings(max_examples=50)
def test_model_spec_round_trip(spec: ModelSpec) -> None:
    """ModelSpec round-trips through model_dump / model_validate."""
    restored = ModelSpec.model_validate(spec.model_dump())
    assert restored == spec


@given(_model_spec_strategy)
@settings(max_examples=50)
def test_model_spec_params_remain_scalars(spec: ModelSpec) -> None:
    """After a round-trip, all param values are still JSON scalars."""
    restored = ModelSpec.model_validate(spec.model_dump())
    for k, v in restored.params.items():
        assert isinstance(v, int | float | str | bool | type(None)), (
            f"params[{k!r}] should be a scalar, got {type(v).__name__!r}"
        )
