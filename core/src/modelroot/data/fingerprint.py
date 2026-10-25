"""Deterministic fingerprints for datasets and splits (M3).

A fingerprint is a stable hex string derived from content, suitable for use
as a cache key or reproducibility token.  It is NOT a cryptographic hash —
security is not required; stability across processes and Python versions is.
"""

from __future__ import annotations

import hashlib

import pandas as pd


def dataset_fingerprint(df: pd.DataFrame) -> str:
    """Compute a stable SHA-256 hex fingerprint of a DataFrame.

    The fingerprint is derived from the column names and the raw bytes of
    each column's values in row order.  It is stable across processes
    provided the DataFrame content is identical.

    Args:
        df: The DataFrame to fingerprint.  Must not be modified concurrently.

    Returns:
        64-character lower-case hexadecimal SHA-256 digest.
    """
    h = hashlib.sha256()
    # Hash column names first so column-rename changes the fingerprint
    h.update("|".join(str(c) for c in df.columns).encode("utf-8"))
    h.update(b"\n")
    for col in df.columns:
        h.update(str(col).encode("utf-8"))
        h.update(b":")
        # Convert each column to string representation for stable hashing
        col_bytes = df[col].astype(str).str.cat(sep="\x00").encode("utf-8")
        h.update(col_bytes)
        h.update(b"\n")
    return h.hexdigest()
