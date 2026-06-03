"""v0.5 analysis-axis enums (§4.1 of refactor_api.md).

Two orthogonal user-facing axes describe an analysis cell:

- ``FactorScope`` — does the factor vary per-asset (``INDIVIDUAL``) or
  carry a single value broadcast to every asset (``COMMON``)?
- ``FactorDensity`` — continuous numeric exposure (``DENSE``) vs.
  ``{0, R}`` event triggers (``SPARSE`` — zero on non-event entries,
  arbitrary real magnitude otherwise; canonical example ``{-1, 0, +1}``)?

``DataStructure`` is the third axis used by registry keys / dispatch but is not
user-set: it is derived from ``N`` at evaluate-time and surfaced as the
third position of ``EvaluationResult.cell`` for downstream pattern-match.
"""

from __future__ import annotations

from enum import StrEnum


class FactorScope(StrEnum):
    """Does each asset have its own factor value, or do all share one?"""

    INDIVIDUAL = "individual"
    COMMON = "common"


class FactorDensity(StrEnum):
    """Continuous numeric exposure vs. ``{0, R}`` sparse event trigger.

    Sparse columns are zero on non-event entries with arbitrary real
    magnitude otherwise (canonical example ``{-1, 0, +1}``).
    """

    DENSE = "dense"
    SPARSE = "sparse"


class DataStructure(StrEnum):
    """Sample regime, derived from ``N`` at evaluate-time.

    ``PANEL``      — ``N >= 2`` (multi-asset / multi-event panel).
    ``TIMESERIES`` — ``N == 1`` (single-asset time series).
    """

    PANEL = "panel"
    TIMESERIES = "timeseries"


class Visibility(StrEnum):
    """Whether a metric appears in user-facing discovery surfaces.

    ``PUBLIC``   — surfaced by :func:`factrix.list_metrics`,
    :attr:`PanelInspection.metrics.applicable`, and
    :attr:`EvaluationResult.metrics` dict keys.
    ``INTERNAL`` — stage-1 helper (intermediate-frame producer such as
    ``compute_ic``); pulled by the DAG executor via
    :attr:`MetricSpec.requires` but not listed as a runnable metric.
    """

    PUBLIC = "public"
    INTERNAL = "internal"
