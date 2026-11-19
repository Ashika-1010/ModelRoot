"""Benchmark grid generator and configuration loader (M13).

Generates deterministic lists of :class:`ScenarioSpec` instances from grid
definitions (e.g. ``benchmark/configs/mini.json``).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from modelroot_bench.scenario import InjectionSpec, ScenarioSpec


def _scenario_id(
    dataset: str,
    model: str,
    scenario_class: str,
    injections: list[InjectionSpec],
    seed: int,
) -> str:
    """Generate a deterministic, legible scenario ID."""
    if not injections or scenario_class == "clean_control":
        return f"{dataset}_cc_{model}_s{seed}"

    parts = [dataset]
    for inj in injections:
        label_clean = inj.label.replace(":", "_")
        sev_clean = inj.severity or "default"
        parts.append(f"{label_clean}_{sev_clean}")
    parts.append(f"{model}_s{seed}")
    return "_".join(parts)


def build_grid(grid_cfg: dict[str, Any]) -> list[ScenarioSpec]:
    """Expand a grid configuration dictionary into a list of ScenarioSpecs.

    Args:
        grid_cfg: Dictionary with a 'scenarios' list.

    Returns:
        List of :class:`ScenarioSpec` instances.
    """
    specs: list[ScenarioSpec] = []
    scenarios_list = grid_cfg.get("scenarios", [])

    for entry in scenarios_list:
        dataset = entry["dataset"]
        model_name = entry.get("model", "random_forest")
        model_params = entry.get("params", {})
        scenario_class = entry.get("scenario_class", "single_failure")
        seed = int(entry.get("seed", 0))
        role = entry.get("role", "dev")

        raw_injections = entry.get("injections", [])
        injections: list[InjectionSpec] = [
            InjectionSpec(
                label=inj["label"],
                severity=inj.get("severity", ""),
                params=inj.get("params", {}),
            )
            for inj in raw_injections
        ]

        scen_id = entry.get("id") or _scenario_id(
            dataset=dataset,
            model=model_name,
            scenario_class=scenario_class,
            injections=injections,
            seed=seed,
        )

        model_spec = {
            "algorithm": model_name,
            "params": model_params,
        }

        spec = ScenarioSpec(
            id=scen_id,
            dataset=dataset,
            model_spec=model_spec,
            scenario_class=scenario_class,
            injections=injections,
            seed=seed,
            role=role,
        )
        specs.append(spec)

    return specs


def load_grid_config(config_path_or_name: str | Path) -> dict[str, Any]:
    """Load a grid configuration by path or predefined name ('mini').

    Args:
        config_path_or_name: Path to a JSON config, or preset name ('mini').

    Returns:
        Parsed configuration dictionary.
    """
    path = Path(config_path_or_name)
    if not path.is_file():
        # Look in benchmark/configs/<name>.json
        repo_bench = Path(__file__).resolve().parent.parent.parent
        candidate = repo_bench / "benchmark" / "configs" / f"{config_path_or_name}.json"
        if candidate.is_file():
            path = candidate
        else:
            local_cand = Path("benchmark") / "configs" / f"{config_path_or_name}.json"
            if local_cand.is_file():
                path = local_cand
            else:
                raise FileNotFoundError(f"Grid configuration not found: {config_path_or_name}")

    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = json.load(f)
        return data
