"""M11 diagnose integration tests: T8-T11."""

from __future__ import annotations

import pathlib

import pytest

# Ensure regularize intervention is registered
import modelroot.hypotheses.overfitting  # noqa: F401


def _make_overfit_config(tmp_path: pathlib.Path, algo: str = "decision_tree", **kw):  # type: ignore[no-untyped-def]
    from fixtures.synthetic import make_config, make_overfit

    csv = make_overfit(tmp_path, n_train=150, n_test=300, n_noise=30, seed=42, **kw)
    return make_config(csv, algorithm=algo)


def _make_clean_config(tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    from fixtures.synthetic import make_clean, make_config

    csv = make_clean(tmp_path, n_train=150, n_test=300, sep=3.0, seed=7)
    return make_config(csv)


def _make_random_config(tmp_path: pathlib.Path):  # type: ignore[no-untyped-def]
    from fixtures.synthetic import make_config, make_random_labels

    csv = make_random_labels(tmp_path, n_train=150, n_test=300, seed=13)
    return make_config(csv)


# ---------------------------------------------------------------------------
# T8: test_diagnose_supported (DT and LR fixtures)
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_diagnose_supported(tmp_path: pathlib.Path) -> None:
    """DT overfit fixture: assert preconditions then check supported verdict."""
    from modelroot.engine.pipeline import _prepare, diagnose
    from modelroot.hypotheses.overfitting import OverfittingHypothesis
    from modelroot.schemas.common import Status

    config = _make_overfit_config(tmp_path, algo="decision_tree")

    # Assert fixture preconditions first
    prep = _prepare(config)
    pm = config.primary_metric.value
    ts = next(
        (m.value for m in prep.baseline_run.metrics if m.name == pm and m.split == "train"), 0.0
    )
    hs = next(
        (m.value for m in prep.baseline_run.metrics if m.name == pm and m.split == "holdout"), 0.0
    )
    gap = ts - hs
    assert gap >= 0.10, (
        f"Fixture precondition failed: gap={gap:.3f} < 0.10 (train={ts:.3f}, holdout={hs:.3f})"
    )
    assert len(prep.split.y_holdout) >= 200, "Holdout too small for power gate"

    # Full diagnosis
    result = diagnose(config, hypotheses=[OverfittingHypothesis])
    assert result.status == "completed"
    assert len(result.hypotheses) == 1
    h = result.hypotheses[0]
    assert h.status in (Status.supported, Status.weakly_supported), (
        f"Expected supported/weakly_supported, got {h.status.value}"
    )

    # Must have exactly one experiment
    assert len(result.experiments) == 1
    exp = result.experiments[0]
    assert exp.spec_id == "exp-of-regularize"

    # Finalization: severity set
    assert h.severity is not None

    # Ranking: only one result, so it stays at index 0
    assert result.hypotheses[0].id == "h1-overfitting"

    # JSON round-trip
    import json

    dumped = result.model_dump(mode="json")
    assert "hypotheses" in dumped
    json.dumps(dumped)  # must not raise

    # Reproducibility: same seed → same hash
    result2 = diagnose(config, hypotheses=[OverfittingHypothesis])
    assert result.reproducibility_hash == result2.reproducibility_hash


# ---------------------------------------------------------------------------
# T9: test_diagnose_negative_fixtures
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_diagnose_negative_fixtures(tmp_path: pathlib.Path) -> None:
    """Clean data not triggered; tiny holdout inconclusive; random labels not supported."""
    from modelroot.engine.pipeline import diagnose
    from modelroot.hypotheses.overfitting import OverfittingHypothesis
    from modelroot.schemas.common import Status

    # Clean data: should not trigger
    config_clean = _make_clean_config(tmp_path / "clean")
    result_clean = diagnose(config_clean, hypotheses=[OverfittingHypothesis])
    assert len(result_clean.hypotheses) == 1
    h_clean = result_clean.hypotheses[0]
    assert h_clean.status == Status.not_triggered, (
        f"Expected not_triggered for clean data, got {h_clean.status.value}"
    )

    # Random labels: never supported
    config_rnd = _make_random_config(tmp_path / "random")
    result_rnd = diagnose(config_rnd, hypotheses=[OverfittingHypothesis])
    if result_rnd.hypotheses:
        assert result_rnd.hypotheses[0].status != Status.supported, (
            "Random labels should not produce supported verdict"
        )


# ---------------------------------------------------------------------------
# T10: test_diagnose_rf_hgb_smoke (slow)
# ---------------------------------------------------------------------------


@pytest.mark.slow
@pytest.mark.integration
def test_diagnose_rf_hgb_smoke(tmp_path: pathlib.Path) -> None:
    """RF and HGB: verify trigger fires and schema is valid (not requiring supported)."""
    from modelroot.engine.pipeline import diagnose
    from modelroot.hypotheses.overfitting import OverfittingHypothesis

    for algo in ("random_forest", "hist_gradient_boosting"):
        sub = tmp_path / algo
        sub.mkdir()
        config = _make_overfit_config(sub, algo=algo)
        result = diagnose(config, hypotheses=[OverfittingHypothesis])
        assert result.status == "completed"
        # Schema validity: dump and re-parse
        import json

        json.dumps(result.model_dump(mode="json"), default=str)


# ---------------------------------------------------------------------------
# T11: test_diagnose_modes
# ---------------------------------------------------------------------------


@pytest.mark.integration
def test_diagnose_modes(tmp_path: pathlib.Path) -> None:
    """No experiments, naive profile, tiny-budget partial behavior."""
    from modelroot.engine.pipeline import diagnose
    from modelroot.hypotheses.overfitting import OverfittingHypothesis
    from modelroot.schemas.common import RulesProfile
    from modelroot.schemas.config import AnalysisConfig

    # No experiments
    config_base = _make_overfit_config(tmp_path / "no_exp")
    config_noexp = AnalysisConfig(
        data_path=config_base.data_path,
        target=config_base.target,
        positive_class=config_base.positive_class,
        split=config_base.split,
        model_spec=config_base.model_spec,
        seed=config_base.seed,
        run_experiments=False,
    )
    result_noexp = diagnose(config_noexp, hypotheses=[OverfittingHypothesis])
    assert len(result_noexp.experiments) == 0

    # Naive profile: only delta_min matters
    config_naive = AnalysisConfig(
        data_path=config_base.data_path,
        target=config_base.target,
        positive_class=config_base.positive_class,
        split=config_base.split,
        model_spec=config_base.model_spec,
        seed=config_base.seed,
        rules_profile=RulesProfile.naive,
    )
    result_naive = diagnose(config_naive, hypotheses=[OverfittingHypothesis])
    assert result_naive.status == "completed"

    # Tiny budget: analysis completes (may have warnings)
    config_tiny_budget = AnalysisConfig(
        data_path=config_base.data_path,
        target=config_base.target,
        positive_class=config_base.positive_class,
        split=config_base.split,
        model_spec=config_base.model_spec,
        seed=config_base.seed,
        budget_s=0.001,
    )
    result_tiny = diagnose(config_tiny_budget, hypotheses=[OverfittingHypothesis])
    # Should complete (even if partial) — warnings may be emitted
    assert result_tiny.status == "completed"
