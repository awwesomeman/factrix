"""``NeweyWest`` reference Estimator instance (#170).

Names the Newey-West HAC inference path that v0.5 cells already emit
into ``FactorProfile.stats``. Carries no compute logic — the underlying
NW HAC math lives in :mod:`factrix._stats` and is invoked by each cell
procedure during :func:`factrix.evaluate`.

The Bartlett kernel + NW1994 auto-bandwidth + Hansen-Hodrick overlap
floor convention applies uniformly across the four primary_p-emitting
cells; cell-specific sample-size guards (e.g. ``UNRELIABLE_SE_SHORT_PERIODS``)
are tracked by the procedures and surface via ``FactorProfile.warnings``.

After #187's StatCode flattening, ``emits_for`` is cell-agnostic — every
applicable cell looks up the same ``StatCode.P`` key. Cell identity is
carried by ``profile.config`` rather than by the StatCode.
"""

from __future__ import annotations

from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode


class NeweyWest:
    """Newey-West (1987) HAC SE estimator → t-statistic → two-sided p-value.

    The Bartlett-kernel HAC variance estimate uses the NW1994 automatic
    bandwidth rule with a Hansen-Hodrick overlap floor for forward-return
    regressions, matching the convention v0.5 procedures already use to
    populate ``primary_p`` and ``StatCode.P`` / ``StatCode.T_NW`` entries.

    Cell interpretation comes from ``profile.config`` (``scope`` /
    ``signal`` / ``metric``) — ``StatCode.P`` is the same key for
    correlation-mean tests (IC), event-window mean-effect tests (CAAR),
    cross-asset slope tests (TS β), etc. Downstream readers must consult
    the config to know which null is being rejected; the StatCode is
    deliberately cell-agnostic to keep ``Estimator.emits_for`` from
    re-encoding cell identity.

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
        # NW HAC drives `primary_p` on every user-facing cell, so the
        # estimator applies universally until a cell opts out.
        del scope, signal
        return True

    def emits_for(
        self,
        scope: FactorScope,
        signal: Signal,
        metric: Metric | None,
    ) -> StatCode:
        del scope, signal, metric
        return StatCode.P
