"""Deterministic seed management using NumPy SeedSequence (M3).

All randomness in ModelRoot is seeded through this module.  A root seed
integer is passed to ``SeedManager``, and each named component obtains its
own independent integer seed via ``spawn(name)``.

Key properties:
- Deterministic: same root + same name → same seed, always.
- Stable across Python processes and versions (no ``hash()``).
- Independent: different names produce independent seeds.
- Collision-resistant: name is hashed with SHA-256 before use.
"""

from __future__ import annotations

import hashlib
import struct

import numpy as np


class SeedManager:
    """Provides deterministic, named seeds derived from a single root seed.

    Args:
        root: The master random seed for this analysis run.  Must be ≥ 0.

    Example::

        sm = SeedManager(42)
        seed_a = sm.spawn("bootstrap")
        seed_b = sm.spawn("split")
        # seed_a and seed_b are stable across runs with the same root.
    """

    def __init__(self, root: int) -> None:
        """Initialise with a non-negative root seed.

        Args:
            root: Master random seed.  Must be ≥ 0.

        Raises:
            ValueError: If ``root`` is negative.
        """
        if root < 0:
            raise ValueError(f"Root seed must be >= 0, got {root}")
        self._root = root
        # Keep a SeedSequence for generating child sequences
        self._seq = np.random.SeedSequence(root)

    def spawn(self, name: str) -> int:
        """Return a deterministic integer seed for *name*.

        The seed is derived by hashing *name* with SHA-256 to obtain a stable
        64-bit integer, then using that integer to spawn a child
        ``SeedSequence`` from the root sequence.  Python's built-in ``hash()``
        is intentionally NOT used because it is randomised across processes.

        Args:
            name: A stable identifier for the component requesting a seed
                (e.g. ``"bootstrap"``, ``"split"``, ``"of:regularize"``).

        Returns:
            A non-negative integer suitable as a ``random_state`` argument for
            scikit-learn estimators.
        """
        # SHA-256 of the name → first 8 bytes → unsigned 64-bit integer
        digest = hashlib.sha256(name.encode("utf-8")).digest()
        name_int = struct.unpack(">Q", digest[:8])[0]  # big-endian uint64

        # Spawn a child sequence seeded by name_int, then generate one seed
        mixed_seq = np.random.SeedSequence([self._root, name_int])
        # Generate a 32-bit seed (scikit-learn uses 32-bit random_state)
        seed_val = int(mixed_seq.generate_state(1)[0])
        return seed_val & 0x7FFF_FFFF  # keep non-negative, fits int32
