"""v0.5 analysis-axis enums (§4.1 of refactor_api.md).

Three orthogonal user-facing axes describe an analysis cell:

- ``FactorScope``  — does the factor vary per-asset (``INDIVIDUAL``) or
  carry a single value broadcast to every asset (``COMMON``)?
- ``Signal``       — continuous numeric exposure (``CONTINUOUS``) vs.
  ``{-1, 0, +1}`` event triggers (``SPARSE``)?
- ``Metric``       — procedure-canonical scalar (``IC`` or ``FM``).
  Only meaningful for ``INDIVIDUAL × CONTINUOUS``; ``None`` for the
  remaining cells.

``Mode`` is the fourth axis used by registry keys / dispatch but is not
user-set: it is derived from ``N`` at evaluate-time and surfaced as
``Profile.mode`` for downstream pattern-match.
"""

from __future__ import annotations

from enum import StrEnum


class FactorScope(StrEnum):
    """Does each asset have its own factor value, or do all share one?"""

    INDIVIDUAL = "individual"
    COMMON = "common"


class Signal(StrEnum):
    """Continuous numeric exposure vs. ``{-1, 0, +1}`` event trigger."""

    CONTINUOUS = "continuous"
    SPARSE = "sparse"


class Metric(StrEnum):
    """Procedure-canonical scalar for ``INDIVIDUAL × CONTINUOUS`` cells."""

    IC = "ic"
    FM = "fm"


class Mode(StrEnum):
    """Sample regime, derived from ``N`` at evaluate-time.

    ``PANEL``      — ``N >= 2`` (multi-asset / multi-event panel).
    ``TIMESERIES`` — ``N == 1`` (single-asset time series).
    """

    PANEL = "panel"
    TIMESERIES = "timeseries"
