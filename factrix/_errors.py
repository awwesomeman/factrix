"""v0.5 exception hierarchy (§4.5).

Three concrete ``ConfigError`` subclasses cover every config-validation
or evaluate-time failure mode; ``suggested_fix`` carries the nearest
legal :class:`factrix._analysis_config.AnalysisConfig` (when one exists)
so the user — or a calling AI agent — can recover programmatically.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig


class FactrixError(Exception):
    """Base for all factrix-raised errors."""


class ConfigError(FactrixError):
    """Base for ``AnalysisConfig`` validation / dispatch errors.

    ``suggested_fix`` is populated from ``_FALLBACK_MAP`` when the
    nearest legal cell is unambiguous (e.g. ``ModeAxisError`` on
    ``(INDIVIDUAL, CONTINUOUS, N=1)`` suggests ``common_continuous``).
    Stays ``None`` when the failure is a data limitation rather than
    an axis-tuple miswire.
    """

    def __init__(
        self,
        message: str,
        *,
        suggested_fix: "AnalysisConfig | None" = None,
    ) -> None:
        super().__init__(message)
        self.suggested_fix = suggested_fix


class IncompatibleAxisError(ConfigError):
    """``(scope, signal, metric)`` tuple is not a legal analysis cell.

    Reachable via direct construction or ``from_dict``; factory methods
    never trigger this. Covers e.g. ``signal=SPARSE`` paired with
    ``metric=IC``, or ``(INDIVIDUAL, CONTINUOUS)`` with ``metric=None``.
    """


class ModeAxisError(ConfigError):
    """Axis tuple is legal but undefined at the runtime ``Mode``.

    Raised at evaluate-time (``Mode`` is not part of ``AnalysisConfig``).
    Canonical example: ``(INDIVIDUAL, CONTINUOUS, IC)`` with ``N == 1``
    has no cross-sectional dispersion → IC undefined; ``suggested_fix``
    points the user at ``common_continuous(...)``.
    """


class InsufficientSampleError(ConfigError):
    """``T < MIN_T_HARD`` for a Mode B procedure.

    Below the floor, NW HAC SE is too biased for ``primary_p`` to be
    trustworthy. Raised at evaluate-time. ``suggested_fix`` is ``None``
    — this is a data limitation, not an axis-tuple miswire. ``actual_T``
    and ``required_T`` carry the numbers so callers can recover or
    aggregate programmatically (review fix UX-3).
    """

    def __init__(
        self,
        message: str,
        *,
        actual_T: int,
        required_T: int,
        suggested_fix: "AnalysisConfig | None" = None,
    ) -> None:
        super().__init__(message, suggested_fix=suggested_fix)
        self.actual_T = actual_T
        self.required_T = required_T
