"""Dataset loading, validation, profiling, splitting, and preprocessing (M2/M3)."""

from __future__ import annotations

from modelroot.data.errors import ContractViolation, ErrorCode
from modelroot.data.fingerprint import dataset_fingerprint
from modelroot.data.loader import Limits, load_csv
from modelroot.data.preprocessing import build_preprocessor
from modelroot.data.profile import profile
from modelroot.data.split import SplitData, make_split
from modelroot.data.validate import ValidatedDataset, validate_for_analysis

__all__ = [
    "ContractViolation",
    "ErrorCode",
    "Limits",
    "SplitData",
    "ValidatedDataset",
    "build_preprocessor",
    "dataset_fingerprint",
    "load_csv",
    "make_split",
    "profile",
    "validate_for_analysis",
]
