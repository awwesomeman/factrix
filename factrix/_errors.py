"""v0.5 exception hierarchy (§4.5).

Three concrete ``ConfigError`` subclasses cover every config-validation
or evaluate-time failure mode; ``suggested_fix`` carries the nearest
legal :class:`factrix._analysis_config.AnalysisConfig` (when one exists)
so the user — or a calling AI agent — can recover programmatically.
"""

from __future__ import annotations

import difflib
from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig


_DOCS_BASE = "https://awwesomeman.github.io/factrix/"


class FactrixError(Exception):
    """Base for all factrix-raised errors."""


class UserInputError(FactrixError):
    """User-supplied input does not match expected names or types.

    Raised for typos in named-set kwargs (metric / p_stat / context key
    / column name) or input-type mismatches. Distinct from
    :class:`ConfigError` (axis miswire) and :class:`InsufficientSampleError`
    (data limitation): catch ``UserInputError`` to handle the
    "user typed the wrong thing" branch separately from internal
    config / sample failures.
    """


def format_user_error(
    *,
    verb: str,
    field: str,
    value: object,
    candidates: Sequence[str] | None = None,
    expected: str | None = None,
    docs_path: str,
) -> str:
    """Render the canonical user-facing error message.

    Exactly one of ``candidates`` (named-set typo) or ``expected``
    (type / shape mismatch) carries the diagnostic; ``candidates`` wins
    when both are passed. ``docs_path`` is appended to
    ``https://awwesomeman.github.io/factrix/`` so the deployed base URL
    is configured in one place.
    """
    if not candidates and not expected:
        raise ValueError("format_user_error requires candidates= or expected=")

    lines: list[str] = []
    if candidates:
        lines.append(f"{verb}(): unknown {field}={value!r}")
        suggestions = difflib.get_close_matches(
            str(value), list(candidates), n=3, cutoff=0.6
        )
        if suggestions:
            quoted = ", ".join(f'"{s}"' for s in suggestions)
            lines.append(f"  Did you mean: {quoted}?")
        lines.append(f"  Available: {sorted(candidates)!r}")
    else:
        lines.append(f"{verb}(): invalid {field}={value!r}")
        lines.append(f"  Expected: {expected}")
    lines.append(f"  Docs: {_DOCS_BASE}{docs_path.lstrip('/')}")
    return "\n".join(lines)


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
        suggested_fix: AnalysisConfig | None = None,
    ) -> None:
        super().__init__(message)
        self.suggested_fix = suggested_fix


class MissingConfigError(ConfigError):
    """``evaluate(raw)`` called without an ``AnalysisConfig``.

    Friendly replacement for the bare ``TypeError`` from the private
    ``_evaluate`` signature. ``suggested_fix`` stays ``None`` — call
    ``factrix.suggest_config(raw)`` to get a concrete recommendation.
    """


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
    """``T < MIN_PERIODS_HARD`` for a TIMESERIES procedure.

    Below the floor, NW HAC SE is too biased for ``primary_p`` to be
    trustworthy. Raised at evaluate-time. ``suggested_fix`` is ``None``
    — this is a data limitation, not an axis-tuple miswire. ``actual_periods``
    and ``required_periods`` carry the numbers so callers can recover or
    aggregate programmatically (review fix UX-3).
    """

    def __init__(
        self,
        message: str,
        *,
        actual_periods: int,
        required_periods: int,
        suggested_fix: AnalysisConfig | None = None,
    ) -> None:
        super().__init__(message, suggested_fix=suggested_fix)
        self.actual_periods = actual_periods
        self.required_periods = required_periods
