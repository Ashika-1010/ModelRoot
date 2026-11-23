"""Failure injector framework (M13)."""

from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.inject.clean import CleanControlInjector
from modelroot_bench.inject.hard_negative_separable import HardNegativeSeparableInjector
from modelroot_bench.inject.imbalance import ImbalanceInjector
from modelroot_bench.inject.overfitting import OverfittingInjector

_INJECTORS: dict[str, type[Injector]] = {
    "clean": CleanControlInjector,
    "clean_control": CleanControlInjector,
    "hard_negative_separable": HardNegativeSeparableInjector,
    "imbalance": ImbalanceInjector,
    "overfitting": OverfittingInjector,
}


def get_injector(name: str) -> Injector:
    """Retrieve an injector instance by name."""
    cls = _INJECTORS.get(name)
    if cls is None:
        raise KeyError(f"Injector {name!r} not registered. Available: {sorted(_INJECTORS.keys())}")
    return cls()


def list_injectors() -> list[str]:
    """Return sorted names of registered injectors."""
    return sorted(_INJECTORS.keys())


__all__ = [
    "CleanControlInjector",
    "HardNegativeSeparableInjector",
    "ImbalanceInjector",
    "InjectedScenario",
    "Injector",
    "OverfittingInjector",
    "get_injector",
    "list_injectors",
]
