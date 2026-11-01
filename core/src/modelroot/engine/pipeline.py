"""High-level pipeline entry points (M6).

Re-exports ``assess_only`` from the assessment verdict module so that
higher-level code only needs to import from ``modelroot.engine``.
"""

from __future__ import annotations

from modelroot.assessment.verdict import assess_only

__all__ = ["assess_only"]
