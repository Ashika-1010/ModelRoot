"""Tests for grid generator and mini-grid configuration (M13)."""

from __future__ import annotations

from modelroot_bench.grid import build_grid, load_grid_config


def test_load_mini_grid_config() -> None:
    cfg = load_grid_config("mini")
    assert cfg["name"] == "mini"
    assert len(cfg["scenarios"]) == 12


def test_build_grid_deterministic_ids() -> None:
    cfg = load_grid_config("mini")
    specs1 = build_grid(cfg)
    specs2 = build_grid(cfg)

    assert len(specs1) == 12
    ids1 = [s.id for s in specs1]
    ids2 = [s.id for s in specs2]
    assert ids1 == ids2
    # Ensure IDs are unique
    assert len(set(ids1)) == len(ids1)


def test_mini_grid_scenario_classes() -> None:
    cfg = load_grid_config("mini")
    specs = build_grid(cfg)
    classes = {s.scenario_class for s in specs}
    assert "clean_control" in classes
    assert "single_failure" in classes
