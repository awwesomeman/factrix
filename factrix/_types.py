"""Shared numerical constants and metric-option aliases.

Axis enums live in :mod:`factrix._axis`. Result dataclasses live in
:mod:`factrix._results`. This module keeps only constants and ``Literal``
aliases shared by metric primitives.
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
#   ``ASSETS``  — cross-sectional asset count (n_assets per date)
#   ``PERIODS`` — time-series length (T, number of dates / draws)
#   ``EVENTS``  — event-date count
# A constant for one axis must never be reused as a threshold on another
# (e.g. an ``_ASSETS`` floor must not gate a series length); introduce a
# separate ``_PERIODS`` constant even when the calibrated value coincides.

# Two-tier per-date asset-count guard for ``compute_ic`` (cross-sectional
# axis). ``MIN_IC_ASSETS_HARD`` is the true computability floor: Spearman IC is
# undefined below two pairwise-complete names. ``MIN_IC_ASSETS_WARN`` is its
# warn-tier counterpart and guards reliability rather than computability: a
# per-period rank correlation over a single-digit cross-section is dominated by
# tie-breaking and sampling noise, so the number is well-defined but carries
# almost no signal. The exact value is a judgement floor, not a simulated one.
# Dates in ``[HARD, WARN)`` are retained but surfaced as thin cross-sections by
# IC consumers and ``inspect_data``.
MIN_IC_ASSETS_HARD: int = 2
MIN_IC_ASSETS_WARN: int = 10

# Minimum sampled series length (periods axis) for sign / mean diagnostics on
# non-overlapping series. Used by IC's post-stride series test, positive_rate's
# binomial sign-count test, and the raw-period base in series-mean inference.
MIN_SERIES_PERIODS_HARD: int = 10

# Two-tier guard for ``directional_hit_rate`` on its ``pairs`` axis — the
# pooled non-overlapping (date, asset) directional trials the
# Pesaran-Timmermann (1992) test treats as independent draws, NOT periods.
# ``n < MIN_DIRECTIONAL_PAIRS_HARD`` short-circuits to NaN (below ~10 pooled
# trials the PT variance estimate var_s = var(P̂) - var(P*) is numerically
# fragile). ``HARD ≤ n < MIN_DIRECTIONAL_PAIRS_WARN`` returns the hit rate
# with ``WarningCode.FEW_DIRECTIONAL_PAIRS`` — the normal approximation to
# S_n is honest only near ~30 pooled trials (the same asymptotic-honesty
# convention as ``MIN_EVENTS_WARN`` / ``MIN_ASSETS_WARN``). Pairs axis, own
# literals — must not alias ``MIN_SERIES_PERIODS_HARD`` (periods axis).
MIN_DIRECTIONAL_PAIRS_HARD: int = 10
MIN_DIRECTIONAL_PAIRS_WARN: int = 30

# Two-tier guard for ``directional_pair_accuracy`` on its ``pairs`` axis: the
# pooled non-overlapping within-date asset pairs whose factor and return
# differences are both non-tied. The metric is descriptive (no p-value), but a
# handful of comparable pairs still makes the per-date ordering read fragile.
# Keep a separate constant from ``MIN_DIRECTIONAL_PAIRS_*`` because those guard
# sign trials, while these guard cross-sectional ordering pairs.
MIN_PAIR_ACCURACY_PAIRS_HARD: int = 10
MIN_PAIR_ACCURACY_PAIRS_WARN: int = 30

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

# Advisory floor on ``estimation_window_event_share`` — the mean, over the
# tested events, of the share of an event's estimation-window bars (or rows)
# that lie inside *another* event's forward-return window on the same asset.
# The mean-adjusted abnormal-return model is identified when that share is
# small: each event's estimation window then measures the asset's
# unconditional mean rather than its neighbours' realised event returns.
# Above this share the per-event abnormal returns are negatively correlated
# through the shared windows and every mean-adjusted event test reads
# conservative (measured sizes in ``_attach_abnormal_return``). Period axis:
# the share counts periods of the window.
ESTIMATION_WINDOW_EVENT_SHARE_WARN: float = 0.25

MIN_OOS_PERIODS_HARD: int = 5

# Two-tier portfolio-period guard for portfolio-level inference (top
# concentration t-test). ``n < MIN_PORTFOLIO_PERIODS_HARD`` short-circuits
# (with 3 dates the cross-time t-test on the per-date ratio is undefined);
# ``HARD ≤ n < WARN`` returns the stat with
# ``WarningCode.BORDERLINE_PORTFOLIO_PERIODS`` and a Python ``UserWarning``.
# Descriptive quantile / asymmetry diagnostics use only HARD.
MIN_PORTFOLIO_PERIODS_HARD: int = 3
MIN_PORTFOLIO_PERIODS_WARN: int = 20

MIN_MONOTONICITY_PERIODS_HARD: int = 5

# Minimum complete (factor, return) observations per asset before
# ``compute_common_betas`` will fit that asset's time-series slope. Counted on
# the panel's period grid, so periods an asset is missing count as missing
# observations and a ragged name can fall under the floor where a dense one
# does not. Assets below it are dropped; the reduction is carried on the
# assets axis by ``_attach_drop_stats`` and surfaced by the cross-asset
# consumers as ``WarningCode.EXCESSIVE_ASSET_DROPS`` once the drop rate clears
# ``DROP_RATE_WARN_THRESHOLD``.
MIN_COMMON_BETA_PERIODS_HARD: int = 20

# Two-tier sample-size guard on the Fama-MacBeth β series. ``T < HARD`` short-
# circuits — a Newey-West HAC SE on a 3-period series is undefined. ``HARD ≤ T
# < WARN`` returns the stat with ``WarningCode.UNRELIABLE_SE_SHORT_PERIODS``
# attached (literature floor: Fama-MacBeth originally used T~30+; below that
# the asymptotic t is borderline). ``T ≥ WARN`` is silent.
MIN_FM_PERIODS_HARD: int = 4
MIN_FM_PERIODS_WARN: int = 30

# Per-date cross-section floor (assets axis) for the robust-scale estimators in
# ``factrix.preprocess.normalize``. Below three finite values a per-date median /
# MAD pair carries no information about dispersion: at n=1 the chain used to
# fall through to ``z = 0.0`` (an "average" fabricated from one observation) and
# at n=2 the MAD-scaled z is ``+-0.6745`` whatever the two values are — a
# constant that is indistinguishable downstream from a real score. Dates below
# the floor are left unscaled (z null, clip skipped) and flagged with
# ``WarningCode.INSUFFICIENT_SCALE_ASSETS``.
MIN_SCALE_ASSETS_HARD: int = 3


# Structural alias used by metric internals to mark "this float is a
# p-value, not an effect-size".
PValue = NewType("PValue", float)


# ---------------------------------------------------------------------------
# Metric-option Literal aliases
# ---------------------------------------------------------------------------

# Clustering correction for CAAR — which within-date (same-period)
# correlation source the Z is built from.
KPSource = Literal["icc", "no_multi_event_dates"]

# Top-bucket concentration weight basis — pure factor magnitude vs.
# realised contribution to the long-leg's α.
ConcentrationWeight = Literal["abs_factor", "alpha_contribution"]

# Quantile-bucketing tie-break policy — how per-period ranks resolve equal
# factor values before they are cut into buckets. ``"ordinal"`` breaks ties by
# row order (balanced bucket sizes, tied names split across buckets);
# ``"average"`` gives tied names a shared average rank (same bucket, possibly
# unbalanced sizes). Both are polars rank methods, but the closed set is
# factrix's: the remaining polars methods (``min`` / ``max`` / ``dense`` /
# ``random``) either distort the bucket widths or make the bucketing
# irreproducible, so they are not part of the contract.
TiePolicy = Literal["ordinal", "average"]

# Canonical sample-dimension vocabulary — the axis a count is measured along.
# Single source of truth for ``MetricResult.n_obs_axis`` and the ``axis`` params
# in ``metrics._helpers`` (drop-stats / floor enforcement); mypy rejects any
# token outside this set, so the grammar (also encoded in ``n_<axis>`` /
# ``min_<axis>`` keys) cannot drift metric-by-metric.
#
# ``pairs`` is the pooled ``(date, asset)`` observation — the unit
# ``pooled_beta`` / ``directional_hit_rate`` treat as an independent draw.
# ``asset_pairs`` is the *within-period* unordered asset couple
# (``C(n_assets, 2)`` per date) that ``directional_pair_accuracy`` orders.
# They are two different units and differ by ~an order of magnitude on the same
# panel, so they must not share a token: a reader stacking ``to_frame`` across
# metrics has only ``n_obs_axis`` to tell one from the other.
SampleAxis = Literal["periods", "events", "pairs", "asset_pairs", "assets"]


# Minimum surplus assets (n_assets - n_base - 1) a per-date cross-sectional OLS
# must leave before ``orthogonalize_factor`` will fit it. On this axis one
# surplus asset is exactly one residual degree of freedom, which is why the
# public knob is spelled ``min_residual_df`` — the statistical term — while the
# constant carries the ASSETS axis token the naming grammar requires.
# The old floor was ``len(base_cols) + 2`` rows, i.e. a single residual df:
# raw R2 is mechanically ~K/(N-1) even at a true R2 of 0, so a 6-name
# cross-section on 4 regressors reported R2 = 0.79 while stripping 83% of the
# factor's variance. Fama-MacBeth practice discards cross-sections with too few
# names per regressor; 10 residual df is the ``N >= K + 10`` form of that rule
# and is exposed as ``min_residual_df`` for callers who want the other
# (``N >= 5K``) convention.
MIN_ORTHOGONALIZE_RESIDUAL_ASSETS: int = 10


# ---------------------------------------------------------------------------
# Shared bucketing / horizon defaults
# ---------------------------------------------------------------------------
#
# One source of truth for the long-short bucketing and the rebalance stride,
# shared by ``quantile_spread``, ``quantile_spread_vw`` and
# ``notional_turnover``. These three are designed to be read together — the
# spread is the gross alpha, the turnover is what it costs to hold — and the
# cost algebra in ``breakeven_cost`` / ``net_spread`` is only valid when the
# spread and the turnover were computed on the *same* bucketing and the *same*
# stride. They previously carried incompatible defaults (n_groups 5 vs 10,
# overlap_periods 5 vs 1), so running each at its own default understated
# breakeven by 5.6x and overstated cost drag by 10.7x on a 60-name panel.
# ``monotonicity`` deliberately keeps its own ``n_groups=10``: a decile curve
# is the shape it is calibrated to read, not a long-short leg.
DEFAULT_N_GROUPS: int = 5
# Coarsest quantile split any bucketing metric accepts. Two groups is the
# top-half / bottom-half long-short book — the split the small-universe
# guidance recommends and the coarsest one where "top" and "bottom" are still
# distinct buckets. One group has no long-short leg at all: the spread is
# identically zero, turnover has nothing to churn and a monotonicity curve is a
# single point. Enforced through one validator (``_validate_n_groups``), called
# at the top of every bucketing entry point before any data work, so every
# consumer — ``quantile_spread``, ``quantile_spread_vw``,
# ``compute_spread_series``, ``monotonicity``, ``notional_turnover`` — rejects
# the same values with the same message instead of each carrying its own
# bound (``notional_turnover`` used to demand three groups while
# ``quantile_spread`` priced the two-group book it could not pair with).
N_GROUPS_FLOOR: int = 2
DEFAULT_FORWARD_PERIODS: int = 5
