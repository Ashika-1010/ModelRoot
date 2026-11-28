"""Failure injector framework (M13/M14/M15/M16)."""

from modelroot_bench.inject.base import InjectedScenario, Injector
from modelroot_bench.inject.clean import CleanControlInjector
from modelroot_bench.inject.hard_negative_highcard import HardNegativeHighcardInjector
from modelroot_bench.inject.hard_negative_separable import HardNegativeSeparableInjector
from modelroot_bench.inject.hard_negative_strong import HardNegativeStrongInjector
from modelroot_bench.inject.imbalance import ImbalanceInjector
from modelroot_bench.inject.leak_duplicate import LeakDuplicateInjector
from modelroot_bench.inject.leak_identifier import LeakIdentifierInjector
from modelroot_bench.inject.leak_proxy import LeakProxyInjector
from modelroot_bench.inject.overfitting import OverfittingInjector
from modelroot_bench.inject.shift import ShiftInjector
from modelroot_bench.inject.subgroup import SubgroupInjector

_INJECTORS: dict[str, type[Injector]] = {
    "clean": CleanControlInjector,
    "clean_control": CleanControlInjector,
    "hard_negative_highcard": HardNegativeHighcardInjector,
    "hard_negative_separable": HardNegativeSeparableInjector,
    "hard_negative_strong": HardNegativeStrongInjector,
    "imbalance": ImbalanceInjector,
    "leak_duplicate": LeakDuplicateInjector,
    "leak_identifier": LeakIdentifierInjector,
    "leak_proxy": LeakProxyInjector,
    "leakage:duplicate": LeakDuplicateInjector,
    "leakage:identifier": LeakIdentifierInjector,
    "leakage:target_proxy": LeakProxyInjector,
    "overfitting": OverfittingInjector,
    "shift": ShiftInjector,
    "subgroup": SubgroupInjector,
}


def register_injector_cls(name: str, cls: type[Injector]) -> None:
    """Register an injector class under a name."""
    _INJECTORS[name] = cls


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
    "HardNegativeHighcardInjector",
    "HardNegativeSeparableInjector",
    "HardNegativeStrongInjector",
    "ImbalanceInjector",
    "InjectedScenario",
    "Injector",
    "LeakDuplicateInjector",
    "LeakIdentifierInjector",
    "LeakProxyInjector",
    "OverfittingInjector",
    "SubgroupInjector",
    "get_injector",
    "list_injectors",
    "register_injector_cls",
]
