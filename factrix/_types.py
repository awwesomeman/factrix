"""Shared types and constants for the v0.4 metric primitives.

The v0.5 axis enums (``FactorScope`` / ``FactorDensity`` / ``DataStructure``)
live in :mod:`factrix._axis`; the v0.5 result type
(``FactorProfile``) lives in :mod:`factrix._profile`. The unified
single-metric result type (``MetricResult``) lives alongside the other
result dataclasses in :mod:`factrix._results`. This module keeps only
the numerical constants and metric-option ``Literal`` aliases shared by
the ``factrix.metrics.*`` primitives that v0.5 procedures wrap.
"""

from __future__ import annotations

from typing import Literal, NewType

# ---------------------------------------------------------------------------
# Numerical constants
# ---------------------------------------------------------------------------

# WHY: shared zero-division floor so std ≈ 0 doesn't inflate t-stat to absurd magnitudes.
EPSILON: float = 1e-9

# WHY: ddof=1 is the sample-std convention (industry standard); fixing it project-wide
# avoids systematic bias across cross-metric comparisons.
# Polars .std() defaults to ddof=1; NumPy requires it explicitly.
DDOF: int = 1

# WHY: 1.4826 = 1/Φ⁻¹(0.75); makes MAD an unbiased estimator of σ under normality.
MAD_CONSISTENCY_CONSTANT: float = 1.4826


# ---------------------------------------------------------------------------
# Minimum sample thresholds (used by metrics primitives)
# ---------------------------------------------------------------------------

# Per-date minimum asset count below which ``compute_ic`` drops the date.
# Renamed from MIN_IC_PERIODS — the "PERIODS" suffix was misleading;
# the value has always been checked against per-date asset counts.
MIN_ASSETS_PER_DATE_IC: int = 10

# Two-tier event-count guard for CAAR / Brown-Warner-family tests.
# ``n < MIN_EVENTS_HARD`` short-circuits to NaN MetricResult (math floor —
# below 4 events the per-event-date series cannot support a meaningful
# t-statistic). ``MIN_EVENTS_HARD ≤ n < MIN_EVENTS_WARN`` returns the
# stat AND emits ``WarningCode.FEW_EVENTS`` so the caller
# knows power is thin (Brown-Warner 1985 convention is ~30 events for
# the asymptotic t to be honest). Descriptive event-quality / horizon /
# clustering metrics use only the HARD floor — they have no formal
# hypothesis test, so the WARN tier would be noise.
MIN_EVENTS_HARD: int = 4
MIN_EVENTS_WARN: int = 30

MIN_OOS_PERIODS: int = 5

# Two-tier portfolio-period guard for portfolio-level inference (top
# concentration t-test). ``n < MIN_PORTFOLIO_PERIODS_HARD`` short-circuits
# (with 3 dates the cross-time t-test on the per-date ratio is undefined);
# ``HARD ≤ n < WARN`` returns the stat with
# ``WarningCode.BORDERLINE_PORTFOLIO_PERIODS`` and a Python ``UserWarning``.
# Descriptive quantile / asymmetry diagnostics use only HARD.
MIN_PORTFOLIO_PERIODS_HARD: int = 3
MIN_PORTFOLIO_PERIODS_WARN: int = 20

MIN_MONOTONICITY_PERIODS: int = 5


# Structural alias used by metric internals to mark "this float is a
# p-value, not an effect-size". Carried for the metrics surface; the
# v0.5 profile schema uses ``primary_p: float`` directly.
PValue = NewType("PValue", float)


# ---------------------------------------------------------------------------
# Metric-option Literal aliases
# ---------------------------------------------------------------------------

# Kolari-Pynnönen (2010) clustering correction for CAAR — which intra-day
# correlation source the Z is built from.
KPSource = Literal["icc", "no_multi_event_dates"]

# Shanken (1992) errors-in-variables correction — which σ²(λ) input feeds
# the second-stage variance inflation.
ShankenVarSource = Literal["user_supplied", "betas_timeseries_proxy"]

# Top-bucket concentration weight basis — pure factor magnitude vs.
# realised contribution to the long-leg's α.
ConcentrationWeight = Literal["abs_factor", "alpha_contribution"]
