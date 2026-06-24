"""Shared types and constants for the v0.4 metric primitives.

The v0.5 axis enums (``FactorScope`` / ``FactorDensity`` / ``DataStructure``)
live in :mod:`factrix._axis`. The unified
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
#
# Naming grammar: ``MIN_[<DOMAIN>_]<AXIS>[_<TIER>]``. The AXIS token is
# mandatory and names the axis the threshold guards:
#   ``ASSETS``  — cross-sectional asset count (N per date)
#   ``PERIODS`` — time-series length (T, number of dates / draws)
#   ``EVENTS``  — event-date count
# A constant for one axis must never be reused as a threshold on another
# (e.g. an ``_ASSETS`` floor must not gate a series length); introduce a
# separate ``_PERIODS`` constant even when the calibrated value coincides.

# Per-date minimum asset count below which ``compute_ic`` drops the date
# (cross-sectional axis). Spearman ρ needs ≥ this many names per date for
# its asymptotic distribution to hold.
MIN_IC_ASSETS: int = 10

# Minimum IC time-series length (periods axis) for a reliable mean / sign
# test on the per-date IC series: the post-stride sample in ``ic()``, the
# series-length floor in ``hit_rate``, and the raw-period base in the
# non-overlapping inference floor all key off this "≥10 independent draws"
# rule. Distinct axis from ``MIN_IC_ASSETS`` despite the shared value — do
# not collapse.
MIN_IC_PERIODS: int = 10

# Two-tier guard for ``directional_hit_rate`` on its ``pairs`` axis — the
# pooled non-overlapping (date, asset) directional trials the
# Pesaran-Timmermann (1992) test treats as independent draws, NOT periods.
# ``n < MIN_DIRECTIONAL_PAIRS_HARD`` short-circuits to NaN (below ~10 pooled
# trials the PT variance estimate var_s = var(P̂) - var(P*) is numerically
# fragile). ``HARD ≤ n < MIN_DIRECTIONAL_PAIRS_WARN`` returns the hit rate
# with ``WarningCode.FEW_DIRECTIONAL_PAIRS`` — the normal approximation to
# S_n is honest only near ~30 pooled trials (the same asymptotic-honesty
# convention as ``MIN_EVENTS_WARN`` / ``MIN_ASSETS_WARN``). Pairs axis, own
# literals — must not alias ``MIN_IC_PERIODS`` (periods axis).
MIN_DIRECTIONAL_PAIRS_HARD: int = 10
MIN_DIRECTIONAL_PAIRS_WARN: int = 30

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

# Canonical sample-dimension vocabulary — the axis a count is measured along.
# Single source of truth for ``MetricResult.n_obs_axis`` and the ``axis`` params
# in ``metrics._helpers`` (drop-stats / floor enforcement); mypy rejects any
# token outside this set, so the grammar (also encoded in ``n_<axis>`` /
# ``min_<axis>`` keys) cannot drift metric-by-metric.
SampleAxis = Literal["periods", "events", "pairs", "assets"]
