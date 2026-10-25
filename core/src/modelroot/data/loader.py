"""CSV loading with safety limits and encoding validation (M2).

Security rules enforced here:
- File-size checked before reading.
- Row-count guard via ``nrows=limit+1`` prevents full-read of giant files.
- Null-byte detection on raw bytes.
- Only UTF-8 / UTF-8-BOM accepted.
- Duplicate column names detected on the raw header line before pandas
  has a chance to silently rename them.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from modelroot.data.errors import ContractViolation, ErrorCode

#: Maximum byte size of a single raw CSV line used when scanning for headers.
_MAX_HEADER_LINE_BYTES = 1_000_000  # 1 MB


@dataclass
class Limits:
    """Resource limits applied when loading a CSV file.

    Attributes:
        max_file_bytes: Maximum allowed file size in bytes.
        max_rows: Maximum allowed number of data rows (excluding header).
        max_cols: Maximum allowed number of columns.
        min_rows: Minimum required number of data rows.
    """

    max_file_bytes: int = 50 * 1024 * 1024  # 50 MB
    max_rows: int = 50_000
    max_cols: int = 50
    min_rows: int = 10


def load_csv(path: str | Path, limits: Limits | None = None) -> pd.DataFrame:
    """Load a CSV file with safety guards and return a clean DataFrame.

    The function applies limits in this order so that expensive operations
    are never started on oversized inputs:

    1. File-size check (before any reads).
    2. Null-byte check (on first ~4 KB of raw bytes).
    3. Encoding check (UTF-8 / UTF-8-BOM only).
    4. Duplicate header detection on the raw first line.
    5. Row-count guard via ``nrows=limit+1``.
    6. Column-count check.
    7. Minimum row-count check.

    Formula-like cells (values starting with ``=``, ``+``, ``-``, ``@``)
    are loaded as plain strings; no spreadsheet-formula interpretation is
    possible inside pandas.

    Args:
        path: Path to the CSV file.
        limits: Resource limits to apply.  Defaults to ``Limits()``.

    Returns:
        A ``pd.DataFrame`` with the raw CSV content.  No type coercions are
        applied; that is left to ``validate_for_analysis``.

    Raises:
        ContractViolation: With one of the specified error codes.
        FileNotFoundError: If the path does not exist.
    """
    if limits is None:
        limits = Limits()

    path = Path(path)

    # ── 1. File-size guard (before any reads) ─────────────────────────────
    file_size = path.stat().st_size
    if file_size > limits.max_file_bytes:
        raise ContractViolation(
            ErrorCode.FILE_TOO_LARGE,
            f"File is {file_size:,} bytes; limit is {limits.max_file_bytes:,} bytes.",
        )

    # ── 2. Read a small prefix to check for null bytes and encoding ────────
    with path.open("rb") as raw:
        prefix = raw.read(4096)

    if b"\x00" in prefix:
        raise ContractViolation(
            ErrorCode.NULL_BYTES,
            "File contains null bytes; binary files (e.g. pickle) are not accepted.",
        )

    # ── 3. Encoding: accept UTF-8 or UTF-8-BOM only ───────────────────────
    try:
        prefix.decode("utf-8-sig")  # accepts both UTF-8 and UTF-8-BOM
    except UnicodeDecodeError:
        raise ContractViolation(
            ErrorCode.BAD_ENCODING,
            "File is not valid UTF-8 or UTF-8-BOM.",
        )

    # Full decode attempt to catch encoding errors beyond the prefix
    try:
        full_text = path.read_bytes().decode("utf-8-sig")
    except UnicodeDecodeError:
        raise ContractViolation(
            ErrorCode.BAD_ENCODING,
            "File contains characters that cannot be decoded as UTF-8.",
        )

    # ── 4. Duplicate header detection on raw line ─────────────────────────
    first_line = full_text.split("\n")[0]
    if not first_line.strip():
        raise ContractViolation(
            ErrorCode.NO_HEADER,
            "File appears to have no header row.",
        )

    # Parse the header line using the csv module to handle quoting correctly
    import csv

    header_reader = csv.reader(io.StringIO(first_line))
    try:
        raw_cols = next(header_reader)
    except StopIteration:
        raise ContractViolation(ErrorCode.NO_HEADER, "Could not parse header row.")

    # Strip whitespace from column names (pandas does not by default)
    stripped = [c.strip() for c in raw_cols]
    if not any(stripped):
        raise ContractViolation(ErrorCode.NO_HEADER, "Header row contains only empty names.")

    # Check for duplicates before pandas can silently rename them
    seen: set[str] = set()
    dupes: list[str] = []
    for col in stripped:
        if col in seen:
            dupes.append(col)
        seen.add(col)
    if dupes:
        raise ContractViolation(
            ErrorCode.DUPLICATE_COLUMNS,
            f"Duplicate column names found: {dupes}",
        )

    # ── 5. Row-count guard: read at most max_rows+1 rows ──────────────────
    # Reading one extra row lets us detect truncation vs. limit exceeded.
    try:
        df_probe = pd.read_csv(
            io.StringIO(full_text),
            nrows=limits.max_rows + 1,
            dtype=str,  # keep everything as str; type inference is in validate_for_analysis
            keep_default_na=False,  # preserve raw strings; NA marking done later
            na_values=[],
            encoding="utf-8",
        )
    except pd.errors.EmptyDataError:
        raise ContractViolation(ErrorCode.NO_HEADER, "File is empty or has no parsable rows.")

    if len(df_probe) > limits.max_rows:
        raise ContractViolation(
            ErrorCode.TOO_MANY_ROWS,
            f"File has more than {limits.max_rows:,} data rows.",
        )

    # ── 6. Column-count check ─────────────────────────────────────────────
    if len(df_probe.columns) > limits.max_cols:
        raise ContractViolation(
            ErrorCode.TOO_MANY_COLS,
            f"File has {len(df_probe.columns)} columns; limit is {limits.max_cols}.",
        )

    # ── 7. Minimum row count ──────────────────────────────────────────────
    if len(df_probe) < limits.min_rows:
        raise ContractViolation(
            ErrorCode.TOO_FEW_ROWS,
            f"File has {len(df_probe):,} data rows; minimum is {limits.min_rows}.",
        )

    return df_probe
