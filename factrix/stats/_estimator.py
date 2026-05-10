"""Estimator protocol — `inference_method` interface for family-verb override (#170).

Family verbs (`bhy` / `bhy_hierarchical` / `partial_conjunction` / `bonferroni`
/ `holm` / `romano_wolf`) accept `estimator: Estimator | None` to select which
inference method's p-value to feed into the step-up math. The protocol carries
*selection* semantics only — the procedure has already populated
``FactorProfile.stats`` with the relevant ``StatCode.*_P`` values; an
``Estimator`` instance names which one to look up for a given cell.

Cell-internal computation (`evaluate()` / standalone metrics) is out of scope:
that future axis goes through a `ComputableEstimator(Estimator)` sub-protocol
that adds a `compute(...)` method.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from factrix._axis import FactorScope, Metric, Signal
    from factrix._codes import StatCode


@runtime_checkable
class Estimator(Protocol):
    """Inference-method instance: names the p-value source family verbs select.

    Implementations supply identity (``name``), human-readable summary
    (``description``), cell-applicability check (``applicable_to``), and a
    cell → StatCode dispatch (``emits_for``) that ``_resolve_family`` uses to
    look up the relevant entry in ``FactorProfile.stats``.

    The protocol is deliberately silent on how the value was originally
    computed — that lives in the procedure that produced the profile.
    """

    @property
    def name(self) -> str:
        """Stable identifier used in error messages and ``list_estimators``."""
        ...

    @property
    def description(self) -> str:
        """One-line summary of the inference method (cell-agnostic).

        Cell-specific stat semantics live in
        ``factrix._codes._STAT_DESCRIPTIONS`` keyed by ``StatCode``.
        """
        ...

    def applicable_to(self, scope: FactorScope, signal: Signal) -> bool:
        """Whether this estimator applies to the ``(scope, signal)`` cell."""
        ...

    def emits_for(
        self,
        scope: FactorScope,
        signal: Signal,
        metric: Metric | None,
    ) -> StatCode:
        """Map a cell to the ``StatCode`` whose value this estimator names.

        Called only after ``applicable_to`` has returned ``True``;
        implementations may assume the cell is in their applicability set.
        """
        ...
