"""Flat exception hierarchy rooted at :class:`FactrixError`.

:class:`IncompatibleAxisError`, :class:`InsufficientSampleError`,
and :class:`UserInputError` each inherit directly from
:class:`FactrixError`; callers can branch on subclass without
parsing message strings.
"""

from __future__ import annotations

import difflib
from collections.abc import Iterable

_DOCS_BASE = "https://awwesomeman.github.io/factrix/"
_VALUE_REPR_CAP = 120
_AVAILABLE_PREVIEW = 15


def _api_docs_path(func_name: str, anchor: str) -> str:
    """Return the deployed API path for a snake-case function name."""
    return f"api/{func_name.replace('_', '-')}#{anchor}"


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

    Raised for typos in named-set kwargs (metric / estimator / params key
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


class IncompatibleAxisError(FactrixError):
    """``(scope, density, metric)`` tuple is not a legal analysis cell.

    Covers e.g. ``density=SPARSE`` paired with ``metric=IC``, or
    ``(INDIVIDUAL, DENSE)`` with ``metric=None``.
    """


class IncompatibleInferenceError(FactrixError):
    """``inference=`` is not in the metric's applicable-inference allowlist.

    Each metric that exposes ``inference=`` declares an
    ``applicable_inference`` frozenset of the methods it actually
    dispatches. Passing anything outside it — a valid ``Inference`` the
    metric does not vet (e.g. ``HansenHodrick`` to ``ic``) or a non-
    ``Inference`` object — raises here instead of silently running an
    unintended test or falling back to the default.

    Structured attributes carry the diagnostic so callers do not parse
    the rendered message:

    - ``func_name``: the calling metric name (no parens)
    - ``value``: the value the caller passed as ``inference``
    - ``applicable``: tuple of allowed inference names for that metric
    """

    def __init__(
        self,
        *,
        func_name: str,
        value: object,
        applicable: Iterable[str],
    ) -> None:
        self.func_name = func_name
        self.value = value
        self.applicable: tuple[str, ...] = tuple(applicable)
        super().__init__(
            f"{func_name}(): inference={_truncate_repr(value)} is not applicable; "
            f"choose one of {list(self.applicable)!r}"
        )


class InsufficientSampleError(FactrixError):
    """``T < MIN_PERIODS_HARD`` for a TIMESERIES procedure.

    Below the floor, Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) SE is too biased for ``primary_p`` to be
    trustworthy. Raised at evaluate-time. ``actual_periods``
    and ``required_periods`` carry the numbers so callers can recover or
    aggregate programmatically (review fix UX-3).
    """

    def __init__(
        self,
        message: str,
        *,
        actual_periods: int,
        required_periods: int,
    ) -> None:
        super().__init__(message)
        self.actual_periods = actual_periods
        self.required_periods = required_periods
