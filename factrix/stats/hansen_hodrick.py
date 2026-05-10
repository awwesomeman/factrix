"""``HansenHodrick`` Estimator instance for the rectangular-kernel HAC path.

Names the Hansen-Hodrick (1980) overlapping-sample inference path
emitted by IC PANEL and FM PANEL procedures into ``profile.stats`` as
``StatCode.P_HH``. Carries no compute logic — the underlying rectangular-
kernel HAC math lives in :mod:`factrix._stats` and is invoked by each
applicable cell procedure during :func:`factrix.evaluate`.

Procedures only populate ``StatCode.P_HH`` when ``forward_periods > 1``;
the ``h = 1`` (non-overlapping) case has no autocovariance terms and HH
collapses to the iid SE — the user is expected to use ``NeweyWest``
there. Calling ``bhy(estimator=HansenHodrick())`` on a non-overlapping
profile lands on a missing-stat error whose message points at the
precondition.
"""

from __future__ import annotations

from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode


class HansenHodrick:
    """Hansen-Hodrick (1980) HAC SE estimator → t-statistic → two-sided p-value.

    Rectangular-kernel HAC variance ``Var(mean) = (γ₀ + 2 Σ_{j=1..h-1}
    γⱼ) / n`` matched to the MA(h-1) overlap structure induced by
    h-period forward returns. No PSD guarantee (Andrews 1991 §3): on
    short / mildly anti-correlated samples the estimate can come out
    negative; the procedure clamps variance to 0 and emits
    ``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE``.

    Applicability is restricted to ``(INDIVIDUAL, CONTINUOUS)`` cells —
    IC PANEL and FM PANEL procedures populate ``StatCode.P_HH``. The
    PANEL-vs-TIMESERIES axis is not part of the protocol; TS β on
    overlapping forward returns is structurally an OLS slope rather than
    a mean t-test, so its HH-OLS variant is deferred to a separate pass.

    Pass an instance to family verbs to make the inference choice
    explicit::

        fx.bhy(profiles, estimator=HansenHodrick())

    Constructor takes no arguments; the kernel and lag-rule are fixed by
    the HH-pure convention.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return (
            "Hansen-Hodrick (1980) rectangular-kernel HAC SE on a mean "
            "(MA(h-1) overlap structure) → t → two-sided p-value."
        )

    def applicable_to(self, scope: FactorScope, signal: Signal) -> bool:
        return scope is FactorScope.INDIVIDUAL and signal is Signal.CONTINUOUS

    def emits_for(
        self,
        _scope: FactorScope,
        _signal: Signal,
        _metric: Metric | None,
    ) -> StatCode:
        return StatCode.P_HH
