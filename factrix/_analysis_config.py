"""v0.5 ``AnalysisConfig`` — three-axis orthogonal factor analysis spec (§4).

The user-facing surface is the four factory methods + ``from_dict`` /
``to_dict``; ``__post_init__`` is the single source of truth for axis
validation, reachable from every path that produces an ``AnalysisConfig``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Self

from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._errors import IncompatibleAxisError
from factrix._registry import matches_user_axis

# Nearest-legal cell suggested when an evaluate-time mode/sample check
# fails (§4.5 A4). Keyed by ``(scope, signal, mode)``; values are
# zero-arg factories so cycles via ``AnalysisConfig`` resolve lazily
# (factory call sites need ``AnalysisConfig`` defined; lazy lambdas
# defer the lookup until raise time, after class definition).
#
# Intentionally narrow — every other legal user triple has a registered
# PANEL *and* TIMESERIES procedure, so ``_evaluate`` never reaches the
# fallback path for them. Only ``(INDIVIDUAL, CONTINUOUS, *)`` lacks a
# TIMESERIES cell (no cross-sectional dispersion at N=1 → IC and per-date
# OLS undefined, §5.5). Adding a TIMESERIES-less cell → add one entry; do
# not encode the suggestion at the ``raise`` site.
_FALLBACK_MAP: dict[
    tuple[FactorScope, Signal, Mode],
    Callable[[], "AnalysisConfig"],
] = {
    (
        FactorScope.INDIVIDUAL,
        Signal.CONTINUOUS,
        Mode.TIMESERIES,
    ): lambda: AnalysisConfig.common_continuous(),
}


def _validate_axis_compat(
    scope: FactorScope,
    signal: Signal,
    metric: Metric | None,
) -> None:
    """Raise ``IncompatibleAxisError`` if the triple is not a legal cell.

    Reverse-queries the registry SSOT (§4.4 A1) — any registered
    ``_DispatchKey`` whose ``(signal, metric)`` matches and whose scope
    either equals ``scope`` or is the collapse sentinel admits the
    triple. Called from ``AnalysisConfig.__post_init__`` so every
    construction path (factory, direct, ``from_dict``) hits one gate.
    """
    if matches_user_axis(scope, signal, metric):
        return
    metric_repr = metric.value if metric is not None else None
    # UX-8 from review: lead with the actionable factory list, leave the
    # tuple enumeration as a parenthetical for users debugging by hand.
    raise IncompatibleAxisError(
        f"({scope.value}, {signal.value}, {metric_repr}) is not a legal "
        "analysis cell. Use one of the four factory methods:\n"
        "  AnalysisConfig.individual_continuous(metric=Metric.IC|Metric.FM)\n"
        "  AnalysisConfig.individual_sparse()\n"
        "  AnalysisConfig.common_continuous()\n"
        "  AnalysisConfig.common_sparse()\n"
        "(legal tuples: (individual, continuous, ic), "
        "(individual, continuous, fm), (individual, sparse, None), "
        "(common, continuous, None), (common, sparse, None).)"
    )


@dataclass(frozen=True, slots=True)
class AnalysisConfig:
    """Three-axis spec for a single-factor analysis.

    Construct via the four factory methods (the supported public API);
    direct construction works but bypasses no validation — every path
    runs through ``__post_init__``.

    Attributes:
        scope: Factor scope axis. ``INDIVIDUAL`` = per-asset factor;
            ``COMMON`` = single broadcast value per date.
        signal: Signal type axis. ``CONTINUOUS`` = real-valued;
            ``SPARSE`` = ``{-1, 0, +1}`` trigger.
        metric: Procedure metric axis. Only populated for
            ``(INDIVIDUAL, CONTINUOUS, *)`` cells (``IC`` or ``FM``);
            ``None`` elsewhere.
        forward_periods: Forward-return horizon in **rows** of the
            panel's time axis, not calendar time. factrix never
            inspects ``date`` dtype or spacing; the caller owns
            frequency and regular spacing. ``forward_periods=5``
            therefore means 5 trading days on a daily panel, 5 weeks
            on a weekly panel, 5 minutes on a 1-min bar panel.
    """

    scope: FactorScope
    signal: Signal
    metric: Metric | None
    forward_periods: int = 5

    def __post_init__(self) -> None:
        _validate_axis_compat(self.scope, self.signal, self.metric)

    @classmethod
    def individual_continuous(
        cls,
        *,
        metric: Metric = Metric.IC,
        forward_periods: int = 5,
    ) -> Self:
        """Per-(date, asset) continuous factor.

        Args:
            metric: ``IC`` for rank predictive ordering; ``FM`` for
                unit-of-exposure premium (Fama-MacBeth λ).
            forward_periods: Forward-return horizon (rows of the time
                axis).

        Returns:
            A validated ``AnalysisConfig`` for the
            ``(INDIVIDUAL, CONTINUOUS, metric)`` cell.
        """
        return cls(
            FactorScope.INDIVIDUAL,
            Signal.CONTINUOUS,
            metric,
            forward_periods=forward_periods,
        )

    @classmethod
    def individual_sparse(cls, *, forward_periods: int = 5) -> Self:
        """Per-(date, asset) sparse trigger (``{-1, 0, +1}``).

        PANEL canonical procedure is the CAAR cross-event t-test;
        TIMESERIES (N=1) collapses to a dummy regression with NW HAC
        SE.

        Args:
            forward_periods: Forward-return horizon (rows of the time
                axis).

        Returns:
            A validated ``AnalysisConfig`` for the
            ``(INDIVIDUAL, SPARSE, None)`` cell.
        """
        return cls(
            FactorScope.INDIVIDUAL,
            Signal.SPARSE,
            None,
            forward_periods=forward_periods,
        )

    @classmethod
    def common_continuous(cls, *, forward_periods: int = 5) -> Self:
        """Broadcast continuous factor (e.g. VIX).

        Canonical procedure is the per-asset β estimate followed by a
        cross-asset t-test on ``E[β]``.

        Args:
            forward_periods: Forward-return horizon (rows of the time
                axis).

        Returns:
            A validated ``AnalysisConfig`` for the
            ``(COMMON, CONTINUOUS, None)`` cell.
        """
        return cls(
            FactorScope.COMMON,
            Signal.CONTINUOUS,
            None,
            forward_periods=forward_periods,
        )

    @classmethod
    def common_sparse(cls, *, forward_periods: int = 5) -> Self:
        """Broadcast sparse trigger (FOMC, policy, index rebalance).

        PANEL canonical: per-asset β on dummy + cross-asset t-test.
        TIMESERIES (N=1): TS dummy regression + NW HAC SE.

        Args:
            forward_periods: Forward-return horizon (rows of the time
                axis).

        Returns:
            A validated ``AnalysisConfig`` for the
            ``(COMMON, SPARSE, None)`` cell.
        """
        return cls(
            FactorScope.COMMON,
            Signal.SPARSE,
            None,
            forward_periods=forward_periods,
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialise to a JSON-compatible dict.

        Returns:
            A dict with string-valued enums and integer
            ``forward_periods``, suitable for JSON serialisation.
        """
        return {
            "scope": self.scope.value,
            "signal": self.signal.value,
            "metric": self.metric.value if self.metric is not None else None,
            "forward_periods": self.forward_periods,
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> Self:
        """Reconstruct from ``to_dict``'s output.

        Goes through ``__post_init__``, so an invalid triple raises
        ``IncompatibleAxisError`` instead of silently constructing.

        Args:
            d: Mapping in the shape produced by ``to_dict``.

        Returns:
            A validated ``AnalysisConfig``.

        Raises:
            IncompatibleAxisError: If the ``(scope, signal, metric)``
                triple is not a legal cell.
        """
        m = d.get("metric")
        return cls(
            scope=FactorScope(d["scope"]),
            signal=Signal(d["signal"]),
            metric=Metric(m) if m is not None else None,
            forward_periods=d.get("forward_periods", 5),
        )
