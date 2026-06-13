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

Metric-spec structural enums (``Aggregation``, ``SpecRole``,
``InputShape``, ``OutputShape``) live here alongside the axis
enums so all enum definitions share one import path.
"""

from __future__ import annotations

from enum import Enum, StrEnum


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


# ---------------------------------------------------------------------------
# Metric-spec structural enums
# ---------------------------------------------------------------------------


class Aggregation(Enum):
    """How the cross-section and time-series reductions compose.

    Load-bearing for the DAG executor and FDR correction — determines
    execution order and which samples are statistically independent.
    """

    CS_THEN_TS = "cs_then_ts"
    """Cross-section step per date → form time series → time aggregation."""

    TS_THEN_CS = "ts_then_cs"
    """Per-asset time-series step → cross-asset aggregation."""

    TS_ONLY = "ts_only"
    """Pure time-series; no cross-asset reduction."""

    EVENT_TIME = "event_time"
    """Event-time stack aggregation (event study / CAAR / rank patterns)."""

    CS_SNAPSHOT = "cs_snapshot"
    """Single cross-section snapshot (static-cs diagnostics such as HHI)."""

    RETURN_SPANNING = "return_spanning"
    """Spanning regression on factor-return time series."""


class SpecRole(Enum):
    """Whether a spec is a user-facing metric or an internal pipeline step.

    ``METRIC``   — public result-producing callable; surfaced by
    :func:`factrix.list_metrics` and included in
    :attr:`EvaluationResult.metrics` dict keys.
    ``PIPELINE`` — stage-1 intermediate-frame producer (e.g.
    ``compute_ic``); pulled by the DAG executor via
    :attr:`MetricSpec.requires` but not listed as a runnable metric.
    """

    METRIC = "metric"
    PIPELINE = "pipeline"


class InputShape(Enum):
    """Shape of data the metric callable directly receives.

    ``PANEL``  — full long-format panel (date × asset_id × factor × return).
    ``SERIES`` — 1-D time series produced by an upstream PIPELINE step.
    ``SCALAR`` — pre-aggregated scalar (e.g. breakeven cost calculation).
    """

    PANEL = "panel"
    SERIES = "series"
    SCALAR = "scalar"


class OutputShape(Enum):
    """Shape of the value the metric callable returns.

    Every ``role=METRIC`` spec must have ``output_shape=SCALAR``
    (enforced by :meth:`MetricSpec.__post_init__`).
    """

    SCALAR = "scalar"
    SERIES = "series"
    PANEL = "panel"


class Tier(StrEnum):
    """Usability tier for a metric on a given panel shape axis.

    Reflects the sample-floor verdict only (cell-match is evaluated
    separately by ``inspect_data``):

    CLEAN: at or above the ``warn`` floor — fully usable, no warning.
    DEGRADED: between the ``min`` and ``warn`` floors — usable but inference
        degraded (e.g. borderline sample size).
    UNUSABLE: below the ``min`` floor — sample too thin to run.
    """

    CLEAN = "clean"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"
