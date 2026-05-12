"""``HansenHodrick`` HAC estimator — rectangular-kernel HAC SE on a series mean.

Names the Hansen-Hodrick (1980) overlapping-sample inference path
emitted to ``profile.stats`` as ``StatCode.P_HH`` / ``StatCode.T_HH``.
``compute(series, *, forward_periods)`` delegates to
:func:`factrix._stats._hansen_hodrick_t_test`.

``forward_periods = 1`` (non-overlapping) has no autocovariance terms
and HH collapses to the iid SE — ``compute`` still delegates to the
primitive, which returns the iid result.
``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE`` surfaces when the
rectangular-kernel sum comes out negative (Andrews 1991 §3).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode, WarningCode
from factrix._stats.constants import MIN_PERIODS_WARN
from factrix.stats._estimator import InferenceResult

if TYPE_CHECKING:
    import numpy as np


class HansenHodrick:
    """Hansen-Hodrick (1980) HAC SE estimator → t-statistic → two-sided p-value.

    Rectangular-kernel HAC variance ``Var(mean) = (γ₀ + 2 Σ_{j=1..h-1}
    γⱼ) / n`` matched to the MA(h-1) overlap structure induced by
    h-period forward returns. No PSD guarantee (Andrews 1991 §3): on
    short / mildly anti-correlated samples the estimate can come out
    negative; ``compute`` clamps variance to 0 and surfaces
    ``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE`` in the result.

    Applicability is restricted to ``(INDIVIDUAL, CONTINUOUS)`` cells —
    IC PANEL and FM PANEL procedures populate ``StatCode.P_HH``. The
    PANEL-vs-TIMESERIES axis is not part of the protocol; TS β on
    overlapping forward returns is structurally an OLS slope rather than
    a mean t-test, so its HH-OLS variant is deferred to a separate pass.

    Pass an instance to ``AnalysisConfig`` to drive evaluate-time
    inference::

        cfg = AnalysisConfig.individual_continuous(
            metric=Metric.IC, estimator=HansenHodrick(),
        )

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

    @property
    def min_periods(self) -> int:
        return MIN_PERIODS_WARN

    def applicable_to(self, scope: FactorScope, signal: Signal) -> bool:
        return scope is FactorScope.INDIVIDUAL and signal is Signal.CONTINUOUS

    def emits_for(
        self,
        _scope: FactorScope,
        _signal: Signal,
        _metric: Metric | None,
    ) -> StatCode:
        return StatCode.P_HH

    def compute(
        self,
        series: np.ndarray,
        *,
        forward_periods: int,
    ) -> InferenceResult:
        from factrix._stats import _hansen_hodrick_t_test

        t_hh, p_hh, _, clamped = _hansen_hodrick_t_test(
            series, forward_periods=forward_periods
        )

        warnings: frozenset[WarningCode] = frozenset()
        if clamped:
            warnings |= frozenset({WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE})
        if 0 < len(series) < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_hh,
            p=p_hh,
            stat_name=StatCode.T_HH,
            p_name=StatCode.P_HH,
            metadata={"kernel": "rectangular", "variance_clamped": clamped},
            warnings=warnings,
        )
