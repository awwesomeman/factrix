"""Shared types and constants for the v0.4 metric primitives.

The v0.5 axis enums (``FactorScope`` / ``Signal`` / ``Metric`` / ``Mode``)
live in :mod:`factrix._axis`; the v0.5 result type
(``FactorProfile``) lives in :mod:`factrix._profile`. This module
keeps only the numerical constants and ``MetricOutput`` shared by the
``factrix.metrics.*`` primitives that v0.5 procedures wrap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Literal, NewType

if TYPE_CHECKING:
    from factrix._metric_index import MetricSpec

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
# Renamed from MIN_IC_PERIODS in #18 — the "PERIODS" suffix was misleading;
# the value has always been checked against per-date asset counts.
MIN_ASSETS_PER_DATE_IC: int = 10

# Two-tier event-count guard for CAAR / Brown-Warner-family tests.
# ``n < MIN_EVENTS_HARD`` short-circuits to NaN MetricOutput (math floor —
# below 4 events the per-event-date series cannot support a meaningful
# t-statistic). ``MIN_EVENTS_HARD ≤ n < MIN_EVENTS_WARN`` returns the
# stat AND emits ``WarningCode.FEW_EVENTS_BROWN_WARNER`` so the caller
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


# ---------------------------------------------------------------------------
# Unified output type for metric primitives
# ---------------------------------------------------------------------------


@dataclass
class MetricOutput:
    """Return type for ``factrix.metrics.*`` primitives.

    Args:
        name: Metric identifier (e.g. "ic_ir", "oos_decay").
        value: Raw metric value.
        n_obs: Sample size the metric primitive's estimator actually
            saw. Same family name as ``FactorProfile.n_obs`` but a
            different scope: per-metric single-stage count, vs. the
            final-stage test denominator at the dispatched-cell level.
            ``None`` for metrics where a single integer count is not
            meaningful (e.g. multi-window CAAR series).
        stat: Test statistic (t, z, W, chi2, ...), when applicable.
        significance: ``***`` / ``**`` / ``*`` / ``""`` derived from
            ``metadata["p_value"]`` when available.
        metadata: Tool-specific context (``p_value``, ``stat_type``,
            ``h0``, ``method`` are the standard keys).
        spec: Back-pointer to the declaring :class:`MetricSpec` from
            the producing module's ``__metric_specs__`` tuple.
            ``None`` for outputs constructed outside the registry
            (free-standing primitive calls, tests, ad-hoc consumers).
            Runners stamp this at dispatch time so downstream code
            (``MetricResultGroup``, serialisers, the DAG executor)
            can recover the spec without a name-keyed lookup.
    """

    name: str
    value: float
    n_obs: int | None = None
    stat: float | None = None
    significance: str | None = None
    metadata: dict[str, object] = field(default_factory=dict)
    spec: MetricSpec | None = None

    def __repr__(self) -> str:
        parts = [f"{self.name}={self.value:.4f}"]
        if self.n_obs is not None:
            parts.append(f"n_obs={self.n_obs}")
        if self.stat is not None:
            parts.append(f"stat={self.stat:.2f}")
        if self.significance:
            parts.append(f"sig={self.significance}")
        return f"MetricOutput({', '.join(parts)})"


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
