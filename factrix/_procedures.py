"""v0.5 ``FactorProcedure`` Protocol + per-cell procedures (§4.4.2 B3).

Each cell maps to exactly one ``FactorProcedure`` instance whose
``compute(raw, config) -> FactorProfile`` is the only place numerical
work happens. Cells whose ``compute`` is not yet wired subclass
``_StubProcedure`` and raise ``NotImplementedError``.

Module-bottom ``register(...)`` calls populate
``factrix._registry._DISPATCH_REGISTRY`` at import time so the
registry SSOT (§4.4 A1) is queryable as soon as the package loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._registry import _SCOPE_COLLAPSED, _DispatchKey, register

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig
    from factrix._profile import FactorProfile


@dataclass(frozen=True, slots=True)
class InputSchema:
    """Raw-data shape contract a procedure expects."""

    required_columns: tuple[str, ...] = ()


@runtime_checkable
class FactorProcedure(Protocol):
    """Pure-compute contract: raw data + config → populated profile."""

    INPUT_SCHEMA: ClassVar[InputSchema]

    def compute(
        self, raw: Any, config: "AnalysisConfig",
    ) -> "FactorProfile": ...


class _StubProcedure:
    """Common base — every method ``raise NotImplementedError``.

    Subclasses set ``INPUT_SCHEMA`` and ``_NAME`` so the error message
    points the caller at the missing implementation cell.
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema()
    _NAME: ClassVar[str] = "FactorProcedure"

    def compute(
        self, raw: Any, config: "AnalysisConfig",
    ) -> "FactorProfile":
        raise NotImplementedError(f"{self._NAME}.compute is not implemented")


class _ICContPanelProcedure:
    """``(INDIVIDUAL, CONTINUOUS, IC, PANEL)`` — per-date Spearman IC.

    Aggregates per-date rank correlations between the factor and the
    forward-return into a time series, then runs an NW HAC t-test on
    its mean (Bartlett kernel, NW1994 automatic lag).
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self, raw: Any, config: "AnalysisConfig",
    ) -> "FactorProfile":
        import numpy as np

        from factrix._codes import StatCode
        from factrix._profile import FactorProfile
        from factrix._stats import _newey_west_t_test, _resolve_nw_lags
        from factrix._stats.constants import auto_bartlett
        from factrix.metrics.ic import compute_ic

        # ``compute_ic`` filters by MIN_IC_PERIODS but does not drop nulls;
        # ``pl.corr`` returns null for zero-variance dates (degenerate
        # factor / tied returns) so the explicit drop is reachable.
        ic_values = compute_ic(raw)["ic"].drop_nulls().to_numpy()
        T = int(len(ic_values))
        # Plan §5.2 picks NW1994 auto_bartlett as the default lag, but
        # h-period forward returns force MA(h-1) structure on the IC
        # series so we floor at ``forward_periods - 1`` (Hansen-Hodrick
        # 1980) to keep the HAC SE consistent. ``_resolve_nw_lags``
        # applies that floor and the ``min(., T-1)`` clip in one place.
        nw_lags = (
            _resolve_nw_lags(T, auto_bartlett(T), config.forward_periods)
            if T >= 2 else 0
        )
        ic_mean = float(np.mean(ic_values)) if T > 0 else 0.0
        t_stat, p_value, _ = _newey_west_t_test(ic_values, lags=nw_lags)

        return FactorProfile(
            config=config,
            mode=Mode.PANEL,
            primary_p=p_value,
            n_obs=T,
            stats={
                StatCode.IC_MEAN: ic_mean,
                StatCode.IC_T_NW: t_stat,
                StatCode.IC_P: p_value,
                StatCode.NW_LAGS_USED: float(nw_lags),
            },
        )


class _FMContPanelProcedure(_StubProcedure):
    _NAME = "_FMContPanelProcedure"


class _CAARSparsePanelProcedure(_StubProcedure):
    _NAME = "_CAARSparsePanelProcedure"


class _CommonContPanelProcedure(_StubProcedure):
    _NAME = "_CommonContPanelProcedure"


class _CommonSparsePanelProcedure(_StubProcedure):
    _NAME = "_CommonSparsePanelProcedure"


class _TSBetaContTimeseriesProcedure(_StubProcedure):
    _NAME = "_TSBetaContTimeseriesProcedure"


class _TSDummySparseTimeseriesProcedure(_StubProcedure):
    """Mode B sparse procedure — shared across both user-facing scopes
    via the ``_SCOPE_COLLAPSED`` sentinel (§5.4.1)."""

    _NAME = "_TSDummySparseTimeseriesProcedure"


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------
# (INDIVIDUAL, CONTINUOUS, *, TIMESERIES) intentionally absent —
# evaluate() raises ModeAxisError there (§5.5).

register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC, Mode.PANEL),
    _ICContPanelProcedure(),
    use_case="Per-date Spearman IC across the asset cross-section.",
    refs=("Grinold (1989)", "Newey & West (1987)"),
)
register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.FM, Mode.PANEL),
    _FMContPanelProcedure(),
    use_case="Fama-MacBeth λ on per-date OLS slope.",
    refs=("Fama & MacBeth (1973)", "Petersen (2009)"),
)
register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.SPARSE, None, Mode.PANEL),
    _CAARSparsePanelProcedure(),
    use_case="Cross-event CAAR with t-test on per-event AR aggregate.",
    refs=("Brown & Warner (1985)", "MacKinlay (1997)"),
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.PANEL),
    _CommonContPanelProcedure(),
    use_case="Per-asset β on broadcast factor + cross-asset t on E[β].",
    refs=("Black, Jensen & Scholes (1972)", "Fama & French (1993)"),
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.SPARSE, None, Mode.PANEL),
    _CommonSparsePanelProcedure(),
    use_case="Per-asset β on broadcast event dummy + cross-asset t.",
    refs=("MacKinlay (1997)",),
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.TIMESERIES),
    _TSBetaContTimeseriesProcedure(),
    use_case="Single-asset OLS β on broadcast factor + NW HAC SE.",
    refs=("Newey & West (1987, 1994)", "Stambaugh (1999)"),
)
register(
    _DispatchKey(_SCOPE_COLLAPSED, Signal.SPARSE, None, Mode.TIMESERIES),
    _TSDummySparseTimeseriesProcedure(),
    use_case="Single-asset calendar-time TS dummy regression + NW HAC SE.",
    refs=("Newey & West (1994)", "Ljung & Box (1978)"),
)
