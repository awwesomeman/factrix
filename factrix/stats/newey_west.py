"""``NeweyWest`` HAC estimator — Bartlett-kernel HAC SE on a series mean.

Names the Newey-West HAC inference path emitted to ``FactorProfile.stats``
as ``StatCode.P_NW`` / ``StatCode.T_NW``. ``compute(series, *,
forward_periods)`` delegates to :func:`factrix._stats._newey_west_t_test`
so cell procedures share one path with the standalone primitive.

The Bartlett kernel + NW1994 auto-bandwidth + Hansen-Hodrick overlap
floor convention applies uniformly across the four primary_p-emitting
cells; ``compute`` pre-resolves the bandwidth via ``_resolve_nw_lags``
to keep the (lags-honest) metadata observable and to match the v0.5
procedure call site bit-for-bit.

``emits_for`` is cell-agnostic — every applicable cell looks up the
same ``StatCode.P_NW`` key (#187 flattened the prefix; #192 added the
``_NW`` algorithm suffix for symmetry with ``T_NW`` / ``P_HH``). Cell
identity is carried by ``profile.config`` rather than by the StatCode.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode, WarningCode
from factrix._stats.constants import MIN_PERIODS_WARN
from factrix.stats._estimator import InferenceResult

if TYPE_CHECKING:
    import numpy as np


class NeweyWest:
    """Newey-West (1987) HAC SE estimator → t-statistic → two-sided p-value.

    The Bartlett-kernel HAC variance estimate uses the NW1994 automatic
    bandwidth rule with a Hansen-Hodrick overlap floor for forward-return
    regressions, matching the convention v0.5 procedures already use to
    populate ``primary_p`` and ``StatCode.P_NW`` / ``StatCode.T_NW`` entries.

    Cell interpretation comes from ``profile.config`` (``scope`` /
    ``signal`` / ``metric``) — ``StatCode.P_NW`` is the same key for
    correlation-mean tests (IC), event-window mean-effect tests (CAAR),
    cross-asset slope tests (TS β), etc. Downstream readers must consult
    the config to know which null is being rejected; the StatCode is
    deliberately cell-agnostic to keep ``Estimator.emits_for`` from
    re-encoding cell identity.

    Pass an instance to ``AnalysisConfig`` to run NW at evaluate time
    (default), or to family verbs to select the NW p-value::

        cfg = AnalysisConfig.individual_continuous(estimator=NeweyWest())
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

    @property
    def min_periods(self) -> int:
        return MIN_PERIODS_WARN

    def applicable_to(self, _scope: FactorScope, _signal: Signal) -> bool:
        # NW HAC drives `primary_p` on every user-facing cell, so the
        # estimator applies universally until a cell opts out.
        return True

    def emits_for(
        self,
        _scope: FactorScope,
        _signal: Signal,
        _metric: Metric | None,
    ) -> StatCode:
        return StatCode.P_NW

    def compute(
        self,
        series: np.ndarray,
        *,
        forward_periods: int,
    ) -> InferenceResult:
        from factrix._stats import _newey_west_t_test, _resolve_nw_lags
        from factrix._stats.constants import auto_bartlett

        n = len(series)
        nw_lags = (
            _resolve_nw_lags(n, auto_bartlett(n), forward_periods) if n >= 2 else 0
        )
        t_stat, p_value, _ = _newey_west_t_test(series, lags=nw_lags)

        warnings: frozenset[WarningCode] = frozenset()
        if 0 < n < self.min_periods:
            warnings = frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_stat,
            p=p_value,
            stat_name=StatCode.T_NW,
            p_name=StatCode.P_NW,
            metadata={"nw_lags": nw_lags},
            warnings=warnings,
        )
