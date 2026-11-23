"""Benchmark scenario runner and ModelRoot diagnosis pipeline executor (M12/M13).

Integrates dataset loading, failure injection, oracle impact calculation,
clean-control baseline execution, and the actual ModelRoot ``diagnose()`` pipeline.
"""

from __future__ import annotations

import tempfile
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from modelroot.engine.pipeline import diagnose
from modelroot.schemas.config import AnalysisConfig, SplitMode

from modelroot_bench.datasets.registry import get_dataset
from modelroot_bench.inject import get_injector
from modelroot_bench.inject.base import InjectedScenario
from modelroot_bench.oracle import oracle_impact
from modelroot_bench.scenario import (
    BenchmarkRecord,
    ScenarioResult,
    ScenarioSpec,
)


def run_scenario(
    spec: ScenarioSpec,
    build_fn: Callable[[ScenarioSpec], Any],
) -> ScenarioResult:
    """Execute a single benchmark scenario using an injected builder or executor.

    Args:
        spec: Specification of the scenario to run.
        build_fn: Callable that accepts *spec* and executes or builds the scenario,
            returning either a :class:`ScenarioResult`, an :class:`AnalysisResult`,
            or a dictionary of execution outputs.

    Returns:
        A validated :class:`ScenarioResult`.
    """
    t0 = time.monotonic()
    raw = build_fn(spec)
    runtime = time.monotonic() - t0

    if isinstance(raw, ScenarioResult):
        if raw.runtime_s <= 0.0:
            return raw.model_copy(update={"runtime_s": runtime})
        return raw

    if isinstance(raw, dict) and "statuses" in raw:
        return ScenarioResult(
            scenario_id=spec.id,
            statuses=dict(raw.get("statuses", {})),
            top1=raw.get("top1"),
            runtime_s=float(raw.get("runtime_s", runtime)),
            analysis_hash=str(raw.get("analysis_hash", "")),
            control_statuses=dict(raw.get("control_statuses", {})),
            effects=dict(raw.get("effects", {})),
            ranking=list(raw.get("ranking", [])),
        )

    if hasattr(raw, "hypotheses") and hasattr(raw, "reproducibility_hash"):
        statuses: dict[str, str] = {}
        effects: dict[str, float] = {}
        ranking: list[str] = []
        top1: str | None = None

        for h in raw.hypotheses:
            h_type = getattr(getattr(h, "type", None), "value", str(getattr(h, "type", "")))
            subtype = getattr(h, "subtype", None)
            if subtype:
                label = str(subtype) if ":" in str(subtype) else f"{h_type}:{subtype}"
            else:
                label = h_type
            status_val = getattr(getattr(h, "status", None), "value", str(getattr(h, "status", "")))
            statuses[label] = status_val
            ranking.append(label)

            if getattr(h, "effect", None) is not None:
                effects[label] = float(getattr(h.effect, "delta", 0.0))

            if top1 is None and status_val in ("supported", "weakly_supported"):
                top1 = label

        return ScenarioResult(
            scenario_id=spec.id,
            statuses=statuses,
            top1=top1,
            runtime_s=runtime,
            analysis_hash=str(raw.reproducibility_hash),
            effects=effects,
            ranking=ranking,
        )

    raise ValueError(f"Unrecognized output from build_fn: {type(raw)}")


def _run_diagnose_pipeline(
    df: Any,
    target: str,
    positive_class: str,
    model_spec: dict[str, Any],
    seed: int,
    budget_s: int = 120,
) -> Any:
    """Helper to run the actual ModelRoot diagnose pipeline on a DataFrame."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        csv_path = Path(tmp_dir) / "scenario.csv"
        df.to_csv(csv_path, index=False)

        if "split" in df.columns:
            split_kwargs = {"mode": SplitMode.split_column, "split_column": "split"}
        else:
            split_kwargs = {"mode": SplitMode.single_file, "test_size": 0.2}

        config = AnalysisConfig(
            data_path=str(csv_path),
            target=target,
            positive_class=positive_class,
            split=split_kwargs,
            model_spec=model_spec,
            seed=seed,
            budget_s=budget_s,
        )

        return diagnose(config)


def execute_scenario(
    spec: ScenarioSpec,
    control_cache: dict[tuple[str, str, int], dict[str, str]] | None = None,
) -> BenchmarkRecord:
    """Execute a scenario end-to-end through the ModelRoot diagnosis pipeline.

    Applies failure injection, measures oracle impact, executes matching clean control,
    runs the full ModelRoot pipeline, and bundles the result into a :class:`BenchmarkRecord`.

    Args:
        spec: Scenario specification.
        control_cache: Optional cache mapping (dataset, model_algo, seed) to clean-control statuses.

    Returns:
        Completed :class:`BenchmarkRecord`.
    """
    if control_cache is None:
        control_cache = {}

    dataset_spec = get_dataset(spec.dataset)
    base_df = dataset_spec.load()

    # Determine injector
    model_spec = (
        dict(spec.model_spec) if isinstance(spec.model_spec, dict) else spec.model_spec.model_dump()
    )

    if not spec.injections or spec.scenario_class == "clean_control":
        injector = get_injector("clean_control")
        injected_scen: InjectedScenario = injector.apply(
            df=base_df,
            dataset_spec=dataset_spec,
            severity="",
            seed=spec.seed,
            model_spec=model_spec,
        )
    else:
        primary_inj = spec.injections[0]
        injector = get_injector(primary_inj.label)
        injected_scen = injector.apply(
            df=base_df,
            dataset_spec=dataset_spec,
            severity=primary_inj.severity,
            seed=spec.seed,
            model_spec=model_spec,
            params=primary_inj.params,
        )
        if primary_inj.expected_subtypes and not injected_scen.ground_truth.expected_subtypes:
            injected_scen.ground_truth.expected_subtypes = dict(primary_inj.expected_subtypes)

    # Oracle impact calculation
    impact, no_effect = oracle_impact(
        clean_df=injected_scen.clean_df,
        scenario_df=injected_scen.df,
        target=dataset_spec.target,
        clean_model_spec=model_spec,
        scenario_model_spec=injected_scen.model_spec,
        seed=spec.seed,
    )
    injected_scen.ground_truth.impact = impact
    injected_scen.ground_truth.no_effect = no_effect

    # Run clean control for adjusted FP if not already a clean control
    algo = str(model_spec.get("algorithm", "random_forest"))
    cache_key = (spec.dataset, algo, spec.seed)

    if spec.scenario_class == "clean_control":
        # The scenario itself is a clean control
        t0 = time.monotonic()
        analysis = _run_diagnose_pipeline(
            df=injected_scen.df,
            target=dataset_spec.target,
            positive_class=dataset_spec.positive,
            model_spec=injected_scen.model_spec,
            seed=spec.seed,
        )
        runtime = time.monotonic() - t0
        res = run_scenario(spec, lambda _: analysis).model_copy(update={"runtime_s": runtime})
        control_cache[cache_key] = dict(res.statuses)
        res = res.model_copy(update={"control_statuses": dict(res.statuses)})
        return BenchmarkRecord(
            spec=spec,
            ground_truth=injected_scen.ground_truth,
            result=res,
        )

    # For non-clean scenarios, check control cache or run clean control
    control_statuses = control_cache.get(cache_key)
    if control_statuses is None:
        clean_injector = get_injector("clean_control")
        clean_scen = clean_injector.apply(
            df=base_df,
            dataset_spec=dataset_spec,
            severity="",
            seed=spec.seed,
            model_spec=model_spec,
        )
        clean_analysis = _run_diagnose_pipeline(
            df=clean_scen.df,
            target=dataset_spec.target,
            positive_class=dataset_spec.positive,
            model_spec=clean_scen.model_spec,
            seed=spec.seed,
        )
        dummy_spec = spec.model_copy(
            update={"id": f"ctrl_{spec.id}", "scenario_class": "clean_control"}
        )
        clean_res = run_scenario(dummy_spec, lambda _: clean_analysis)
        control_statuses = dict(clean_res.statuses)
        control_cache[cache_key] = control_statuses

    # Run actual scenario
    t0 = time.monotonic()
    analysis = _run_diagnose_pipeline(
        df=injected_scen.df,
        target=dataset_spec.target,
        positive_class=dataset_spec.positive,
        model_spec=injected_scen.model_spec,
        seed=spec.seed,
    )
    runtime = time.monotonic() - t0
    res = run_scenario(spec, lambda _: analysis).model_copy(
        update={"runtime_s": runtime, "control_statuses": control_statuses}
    )

    return BenchmarkRecord(
        spec=spec,
        ground_truth=injected_scen.ground_truth,
        result=res,
    )
