"""v0.5 exception hierarchy (§4.5).

Three concrete ``ConfigError`` subclasses cover every config-validation
or evaluate-time failure mode; ``suggested_fix`` carries the nearest
legal :class:`factrix._analysis_config.AnalysisConfig` (when one exists)
so the user — or a calling AI agent — can recover programmatically.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable
from typing import Any

_DOCS_BASE = "https://awwesomeman.github.io/factrix/"
_VALUE_REPR_CAP = 120
_AVAILABLE_PREVIEW = 15


def _truncate_repr(value: object) -> str:
    text = repr(value)
    if len(text) <= _VALUE_REPR_CAP:
        return text
    return text[: _VALUE_REPR_CAP - 3] + "..."


def _render_available(candidates: tuple[str, ...]) -> str:
    if len(candidates) <= _AVAILABLE_PREVIEW:
        return f"  Available: {list(candidates)!r}"
    preview = list(candidates[:_AVAILABLE_PREVIEW])
    return (
        f"  Available ({_AVAILABLE_PREVIEW} of {len(candidates)}, "
        f"see Docs): {preview!r}"
    )


class FactrixError(Exception):
    """Base for all factrix-raised errors."""


class UserInputError(FactrixError, ValueError):
    """User-supplied input does not match expected names or types.

    Raised for typos in named-set kwargs (metric / estimator / context key
    / column name) or input-type mismatches. Multi-inherits from
    :class:`ValueError` so ecosystem code (`pytest.raises(ValueError)`,
    generic `except ValueError`) keeps working.

    Structured attributes carry the diagnostic so callers (sub-issue
    raises, LLM agents) do not parse the rendered message:

    - ``func_name``: the calling function name (no parens)
    - ``field``: the kwarg / column name that failed validation
    - ``value``: the value the caller passed in
    - ``candidates``: tuple of legal names (named-set branch); empty otherwise
    - ``suggestions``: difflib top-3 fuzzy matches against ``candidates``
    - ``expected``: human-readable shape (type-mismatch branch); ``None`` otherwise
    - ``docs_url``: deployed-docs URL for the function

    """

    def __init__(
        self,
        *,
        func_name: str,
        field: str,
        value: object,
        candidates: Iterable[object] | None = None,
        expected: str | None = None,
        docs_path: str,
    ) -> None:
        if not candidates and not expected:
            raise ValueError("UserInputError requires candidates= or expected=")
        ordered = tuple(sorted(str(c) for c in candidates)) if candidates else ()
        suggestions = (
            tuple(difflib.get_close_matches(str(value), ordered, n=3, cutoff=0.6))
            if ordered
            else ()
        )
        self.func_name = func_name
        self.field = field
        self.value = value
        self.candidates: tuple[str, ...] = ordered
        self.suggestions: tuple[str, ...] = suggestions
        self.expected = expected
        self.docs_url = f"{_DOCS_BASE}{docs_path.lstrip('/')}"
        super().__init__(self._render())

    def _render(self) -> str:
        value_repr = _truncate_repr(self.value)
        lines: list[str] = []
        if self.candidates:
            lines.append(f"{self.func_name}(): unknown {self.field}={value_repr}")
            if self.suggestions:
                quoted = ", ".join(f'"{s}"' for s in self.suggestions)
                lines.append(f"  Did you mean: {quoted}?")
            lines.append(_render_available(self.candidates))
        else:
            lines.append(f"{self.func_name}(): invalid {self.field}={value_repr}")
            lines.append(f"  Expected: {self.expected}")
        lines.append(f"  Docs: {self.docs_url}")
        return "\n".join(lines)


class ConfigError(FactrixError):
    """Base for cell / dispatch validation errors.

    ``suggested_fix`` carries any caller-actionable recovery payload
    (e.g. nearest-legal cell); stays ``None`` when the failure is a
    data limitation rather than a cell miswire.
    """

    def __init__(
        self,
        message: str,
        *,
        suggested_fix: Any | None = None,
    ) -> None:
        super().__init__(message)
        self.suggested_fix = suggested_fix


class UnknownEstimatorError(ConfigError, ValueError):
    """``get_estimator(name)`` lookup miss (#163).

    Inherits ``ValueError`` so ``pytest.raises(ValueError)`` and the
    ecosystem ``UserInputError`` convention both catch it. Raised by
    ``factrix.stats.get_estimator`` when ``estimator`` name is not in
    the registry.
    """


class IncompatibleAxisError(ConfigError):
    """``(scope, signal, metric)`` tuple is not a legal analysis cell.

    Covers e.g. ``signal=SPARSE`` paired with ``metric=IC``, or
    ``(INDIVIDUAL, CONTINUOUS)`` with ``metric=None``.
    """


class InsufficientSampleError(ConfigError):
    """``T < MIN_PERIODS_HARD`` for a TIMESERIES procedure.

    Below the floor, Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) SE is too biased for ``primary_p`` to be
    trustworthy. Raised at evaluate-time. ``suggested_fix`` is ``None``
    — this is a data limitation, not a cell miswire. ``actual_periods``
    and ``required_periods`` carry the numbers so callers can recover or
    aggregate programmatically (review fix UX-3).
    """

    def __init__(
        self,
        message: str,
        *,
        actual_periods: int,
        required_periods: int,
        suggested_fix: Any | None = None,
    ) -> None:
        super().__init__(message, suggested_fix=suggested_fix)
        self.actual_periods = actual_periods
        self.required_periods = required_periods
