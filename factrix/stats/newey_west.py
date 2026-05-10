"""``NeweyWest`` reference Estimator instance (#170).

Names the Newey-West HAC inference path that v0.5 cells already emit
into ``FactorProfile.stats``. Carries no compute
logic — the underlying NW HAC math lives in :mod:`factrix._stats` and is
invoked by each cell procedure during :func:`factrix.evaluate`.

The Bartlett kernel + NW1994 auto-bandwidth + Hansen-Hodrick overlap
floor convention applies uniformly across the four primary_p-emitting
cells; cell-specific sample-size guards (e.g. ``UNRELIABLE_SE_SHORT_PERIODS``)
are tracked by the procedures and surface via ``FactorProfile.warnings``.
"""

from __future__ import annotations

from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode

# Cell → emitted p-value StatCode for the NW HAC convention.
# Mode is intentionally absent: the same StatCode is emitted in PANEL
# and TIMESERIES variants of a given (scope, signal, metric) cell.
_EMITS: dict[tuple[FactorScope, Signal, Metric | None], StatCode] = {
    (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC): StatCode.IC_P,
    (FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.FM): StatCode.FM_LAMBDA_P,
    (FactorScope.INDIVIDUAL, Signal.SPARSE, None): StatCode.CAAR_P,
    (FactorScope.COMMON, Signal.CONTINUOUS, None): StatCode.TS_BETA_P,
    (FactorScope.COMMON, Signal.SPARSE, None): StatCode.TS_BETA_P,
}


class NeweyWest:
    """Newey-West (1987) HAC SE estimator → t-statistic → two-sided p-value.

    The Bartlett-kernel HAC variance estimate uses the NW1994 automatic
    bandwidth rule with a Hansen-Hodrick overlap floor for forward-return
    regressions, matching the convention v0.5 procedures already use to
    populate ``primary_p`` and ``StatCode.*_T_NW`` / ``*_P`` entries.

    Pass an instance to family verbs to make the inference choice
    explicit::

        fx.bhy(profiles, estimator=NeweyWest())

    Constructor takes no arguments in this release; lag / kernel /
    overlap-floor knobs are tracked as a future enhancement and would
    arrive as keyword-only ``__init__`` parameters without changing
    callers that use the default.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return (
            "Newey-West HAC SE (Bartlett kernel, NW1994 auto-bandwidth, "
            "Hansen-Hodrick overlap floor) → t → two-sided p-value."
        )

    def applicable_to(self, scope: FactorScope, signal: Signal) -> bool:
        return any((s, sig) == (scope, signal) for (s, sig, _) in _EMITS)

    def emits_for(
        self,
        scope: FactorScope,
        signal: Signal,
        metric: Metric | None,
    ) -> StatCode:
        return _EMITS[(scope, signal, metric)]
