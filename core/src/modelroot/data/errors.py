"""Custom exceptions and error codes for ModelRoot dataset operations (M2)."""

from __future__ import annotations

from enum import Enum


class ErrorCode(str, Enum):
    """Stable error codes for all dataset contract violations.

    These codes are part of the public API.  Do not change values.
    """

    FILE_TOO_LARGE = "FILE_TOO_LARGE"
    TOO_MANY_ROWS = "TOO_MANY_ROWS"
    TOO_MANY_COLS = "TOO_MANY_COLS"
    BAD_ENCODING = "BAD_ENCODING"
    NULL_BYTES = "NULL_BYTES"
    NO_HEADER = "NO_HEADER"
    DUPLICATE_COLUMNS = "DUPLICATE_COLUMNS"
    TARGET_MISSING = "TARGET_MISSING"
    TARGET_NOT_BINARY = "TARGET_NOT_BINARY"
    TARGET_HAS_NULLS = "TARGET_HAS_NULLS"
    TOO_FEW_ROWS = "TOO_FEW_ROWS"
    CLASS_TOO_SMALL = "CLASS_TOO_SMALL"
    POSITIVE_CLASS_INVALID = "POSITIVE_CLASS_INVALID"
    # M3 addition (used in split.py, defined here to keep error codes centralised)
    CURRENT_SCHEMA_MISMATCH = "CURRENT_SCHEMA_MISMATCH"


class ContractViolation(Exception):
    """Raised when an input dataset or configuration violates a contract.

    Attributes:
        code: Stable machine-readable error code.
        message: Human-readable description of the violation.
    """

    def __init__(self, code: ErrorCode | str, message: str) -> None:
        """Initialise a ContractViolation.

        Args:
            code: Stable error code identifying the violation type.
            message: Human-readable description of the problem.
        """
        self.code: ErrorCode = ErrorCode(code) if isinstance(code, str) else code
        self.message = message
        super().__init__(f"[{self.code.value}] {message}")
