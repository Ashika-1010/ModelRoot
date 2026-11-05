"""Budget tracking for the experiment runner (M9).

``BudgetTracker`` tracks elapsed wall-clock time against a budget.  It is
checked at cooperative checkpoints; synchronous operations cannot be
forcibly interrupted, so a budget overrun is bounded by one operation.
"""

from __future__ import annotations

import time


class BudgetExceeded(Exception):
    """Raised when the analysis budget has been exhausted.

    Synchronous operations cannot be forcibly interrupted; a budget
    overrun is bounded by the duration of the single operation that
    exceeded the limit.
    """


class BudgetTracker:
    """Wall-clock budget tracker for the experiment runner.

    Args:
        budget_s: Total budget in seconds (> 0).

    Raises:
        ValueError: If *budget_s* <= 0.
    """

    def __init__(self, budget_s: float) -> None:
        if budget_s <= 0:
            raise ValueError(f"budget_s must be > 0, got {budget_s}")
        self._budget_s = budget_s
        self._start = time.monotonic()

    @property
    def elapsed_s(self) -> float:
        """Seconds since the tracker was created."""
        return time.monotonic() - self._start

    @property
    def remaining_s(self) -> float:
        """Seconds remaining in the budget (may be negative)."""
        return self._budget_s - self.elapsed_s

    def check(self) -> None:
        """Raise :class:`BudgetExceeded` if the budget has elapsed.

        This is a cooperative checkpoint.  Callers should call it
        between experiments, not inside tight inner loops.

        Raises:
            BudgetExceeded: If ``elapsed_s >= budget_s``.
        """
        if self.elapsed_s >= self._budget_s:
            raise BudgetExceeded(
                f"Budget of {self._budget_s:.1f}s exceeded (elapsed {self.elapsed_s:.1f}s)."
            )

    def child(self, budget_s: float) -> BudgetTracker:
        """Create a child tracker with a smaller per-experiment budget.

        The child budget is capped to the parent's remaining time so it
        never exceeds the overall analysis budget.

        Args:
            budget_s: Desired per-experiment budget in seconds.

        Returns:
            A new :class:`BudgetTracker`.
        """
        effective = min(budget_s, max(0.001, self.remaining_s))
        return BudgetTracker(effective)
