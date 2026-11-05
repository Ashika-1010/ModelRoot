"""In-memory prediction cache for the experiment runner (M9).

The cache stores baseline ``PredictionSet`` objects keyed by a hash of
(dataset_fingerprint, split_fingerprint, spec_hash, seed, metric) so that
experiments sharing the same baseline avoid redundant re-fitting.

Only successful prediction sets are cached; failures are never stored.
"""

from __future__ import annotations

import hashlib


class PredictionCache:
    """In-memory cache for baseline prediction sets.

    The cache is keyed by a SHA-256 hex digest of the supplied components.
    It stores arbitrary objects (``PredictionSet`` instances in practice).
    """

    def __init__(self) -> None:
        self._store: dict[str, object] = {}

    @staticmethod
    def make_key(
        dataset_fp: str,
        split_fp: str,
        spec_hash: str,
        seed: int,
        metric: str,
    ) -> str:
        """Compute the cache key for a baseline prediction set.

        Args:
            dataset_fp: Dataset fingerprint hex string.
            split_fp: Split fingerprint hex string.
            spec_hash: Hash of the model specification.
            seed: Integer seed used for training.
            metric: Primary metric name.

        Returns:
            64-character lowercase hex SHA-256 digest.
        """
        payload = f"{dataset_fp}|{split_fp}|{spec_hash}|{seed}|{metric}"
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()

    def get(self, key: str) -> object | None:
        """Return the cached object for *key*, or ``None`` if absent.

        Args:
            key: Cache key produced by :meth:`make_key`.

        Returns:
            Cached object or ``None``.
        """
        return self._store.get(key)

    def put(self, key: str, value: object) -> None:
        """Store *value* under *key*.

        Args:
            key: Cache key.
            value: Object to cache (typically a ``PredictionSet``).
        """
        self._store[key] = value

    def __len__(self) -> int:
        return len(self._store)

    def clear(self) -> None:
        """Remove all entries from the cache."""
        self._store.clear()
