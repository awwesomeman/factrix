"""v0.5 ``FactorProcedure`` Protocol + per-cell stubs (§4.4.2 B3).

Each cell maps to exactly one ``FactorProcedure`` instance whose
``compute(raw, config) -> FactorProfile`` is the only place numerical
work happens. This batch ships stubs only — every ``compute`` raises
``NotImplementedError`` with the procedure name. Real implementations
arrive in the next batch (§5.2 / §8.1).

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


class _ICContPanelProcedure(_StubProcedure):
    _NAME = "_ICContPanelProcedure"


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
