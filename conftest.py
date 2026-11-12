"""Root conftest: add core/tests to sys.path so fixtures are importable."""

from __future__ import annotations

import sys
from pathlib import Path

# Make core/tests importable as a package root (for `from fixtures.synthetic import ...`)
_tests_root = Path(__file__).parent / "core" / "tests"
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))
