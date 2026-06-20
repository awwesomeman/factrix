"""v0.5 enum codes for warnings.

``WarningCode`` follows the ``*Code`` suffix invariant (§7.5).
"""

from __future__ import annotations

from enum import StrEnum


class WarningCode(StrEnum):
    """Procedure-degradation flags (replaces v3 ``DegradedMode``).

    Each value carries a one-line ``description`` gloss for
    ``profile.diagnose()`` consumers (review fix UX-4) — pure metadata,
    StrEnum value identity is unchanged.
    """

    UNRELIABLE_SE_SHORT_PERIODS = "unreliable_se_short_periods"
    EVENT_WINDOW_OVERLAP = "event_window_overlap"
    # Fired when ADF p > 0.1 on a DENSE factor (Stambaugh-style
    # persistent-regressor flag, §5.2 / §7.3). Not raised for SPARSE.
    PERSISTENT_REGRESSOR = "persistent_regressor"
    SERIAL_CORRELATION_DETECTED = "serial_correlation_detected"
    # Single cross-asset N guard for PANEL common_continuous: the cross-asset
    # t-test on E[β] runs for any N≥2 (this axis never raises) but its t_crit
    # inflates as N shrinks. One code flags the whole thin regime
    # (n_assets < MIN_ASSETS_WARN); severity is read from the ``n_assets``
    # metadata rather than split across separate tier members.
    FEW_ASSETS = "few_assets"
    # Fired by the (COMMON, SPARSE, PANEL) procedure when the broadcast
    # dummy carries MIN_BROADCAST_EVENTS_HARD ≤ n_events <
    # MIN_BROADCAST_EVENTS_WARN. Per-asset β is identifiable but
    # the cross-event averaging is too thin for asymptotic t to be
    # trusted. Below the HARD floor raises InsufficientSampleError instead.
    SPARSE_COMMON_FEW_EVENTS = "sparse_common_few_events"
    # Fired when a sparse ``factor`` column carries mixed signs but is
    # not a clean ±1 ternary (e.g. ``{-2.5, 0, +1.3}``). The CAAR /
    # sparse-panel statistic is the magnitude-weighted Sefcik-Thompson
    # (1986) variant, which differs from the textbook MacKinlay (1997)
    # signed CAAR at finite samples when negative- and positive-leg
    # vols disagree. ``{-1, 0, +1}`` does not trigger — sign and weight
    # semantics coincide numerically. All-non-negative columns
    # (``{0, 1}`` / ``{0, R≥0}``) do not trigger — no flip ambiguity.
    SPARSE_MAGNITUDE_WEIGHTED = "sparse_magnitude_weighted"
    # Fired by ``caar`` (significance test) when the per-event-date series
    # length sits in ``[MIN_EVENTS_HARD, MIN_EVENTS_WARN)`` — the t-stat
    # is returned but the Brown-Warner (1985) convention treats sub-30
    # event-date counts as power-thin for the asymptotic t-distribution.
    # Below the HARD floor the primitive short-circuits to NaN instead.
    # Naming follows the ``<axis>_<condition>`` grammar; the Brown-Warner
    # method reference lives in this gloss rather than the member name.
    FEW_EVENTS = "few_events"
    # Fired by ``top_concentration`` when the per-date ratio series sits
    # in ``[MIN_PORTFOLIO_PERIODS_HARD, MIN_PORTFOLIO_PERIODS_WARN)`` —
    # the one-sided t-test on the diversification ratio is returned but
    # ``df = n - 1 < 19`` inflates t_crit relative to the asymptotic
    # cutoff. Below the HARD floor the primitive short-circuits to NaN.
    BORDERLINE_PORTFOLIO_PERIODS = "borderline_portfolio_periods"
    # Fired when a rectangular-kernel HAC primitive (Hansen-Hodrick 1980)
    # produces a negative variance-of-mean estimate on short / mildly
    # anti-correlated samples. Unlike the Bartlett kernel, the rectangular
    # kernel carries no PSD guarantee (Andrews 1991 §3); the primitive
    # clamps variance to 0.0 and the t-test returns t=0, p=1.0 (cannot
    # reject), the conservative direction.
    RECT_KERNEL_NEGATIVE_VARIANCE = "rect_kernel_negative_variance"

    # Fired by the DAG executor when an upstream producer short-circuited
    # (returned a NaN MetricResult with metadata["reason"]) and the
    # consumer is skipped. The downstream MetricResult carries
    # metadata["upstream"] / ["upstream_reason"] so the original cause
    # is recoverable without re-walking the dependency graph.
    UPSTREAM_UNAVAILABLE = "upstream_unavailable"

    # Fired by inspect_data when factor columns carry inconsistent axes.
    CROSS_FACTOR_DENSITY_MISMATCH = "cross_factor_density_mismatch"
    CROSS_FACTOR_SCOPE_MISMATCH = "cross_factor_scope_mismatch"

    # Fired by evaluate(strict=False) when a metric's declared
    # cell.structure disagrees with the data's structure (e.g. a PANEL
    # metric requested on TIMESERIES data). The metric is not executed —
    # it short-circuits to a NaN MetricResult with
    # metadata["reason"]="structure_mismatch" rather than computing a
    # numerically real but structurally invalid value.
    STRUCTURE_MISMATCH = "structure_mismatch"

    # Per-axis silent-drop flags. A metric whose upstream primitive silently
    # dropped a large share of its sample at a filter raises the code for the
    # dropped axis: PERIOD_DROPS for the time axis (e.g. compute_ic dropping
    # dates with n_assets below the per-date floor), ASSET_DROPS for the
    # cross-section (e.g. compute_ts_betas dropping assets with insufficient
    # history or zero factor variance). The code is dimension-specific by
    # design — a reader resolves the dropped axis from the code alone, not by
    # digging into metadata (the dimension-naming grammar shared with
    # SampleThreshold and the n_<axis> sample-size constants). A single
    # aggregate flag per metric replaces per-row noise; the exact drop-stat
    # schema (n_<axis>_in / n_<axis>_out / dropped_<axis> / drop_rate /
    # drop_reason) rides in MetricResult.metadata. Fires only when drop_rate
    # exceeds DROP_RATE_WARN_THRESHOLD.
    EXCESSIVE_PERIOD_DROPS = "excessive_period_drops"
    EXCESSIVE_ASSET_DROPS = "excessive_asset_drops"

    @property
    def description(self) -> str:
        return _WARNING_DESCRIPTIONS[self]


_WARNING_DESCRIPTIONS: dict[WarningCode, str] = {}


_WARNING_DESCRIPTIONS.update(
    {
        WarningCode.UNRELIABLE_SE_SHORT_PERIODS: "n_periods is below the WARN floor (~30); NW HAC SE may be biased. "
        "Reused across panel time-series guards (MIN_PERIODS_WARN) and "
        "primitive inference (MIN_FM_PERIODS_WARN); both default to 30.",
        WarningCode.EVENT_WINDOW_OVERLAP: "Adjacent events sit within forward_periods; AR windows overlap.",
        WarningCode.PERSISTENT_REGRESSOR: "ADF p > 0.10 on the continuous factor; β may carry Stambaugh bias.",
        WarningCode.SERIAL_CORRELATION_DETECTED: "Ljung-Box p < 0.05 on residuals; NW lag may be under-set.",
        WarningCode.FEW_ASSETS: "PANEL cross-asset t-test with n_assets < "
        "MIN_ASSETS_WARN (30); df=n_assets-1 inflates t_crit relative to the "
        "asymptotic 1.96 (≈4.30 at n_assets=3, +119%; 5–15% near 30). "
        "Severity scales with n_assets — read the n_assets metadata.",
        WarningCode.SPARSE_COMMON_FEW_EVENTS: "(COMMON, SPARSE, PANEL) broadcast dummy has "
        "MIN_BROADCAST_EVENTS_HARD ≤ n_events < MIN_BROADCAST_EVENTS_WARN "
        "(5..19); per-asset β estimable but cross-event averaging too thin "
        "for asymptotic t.",
        WarningCode.SPARSE_MAGNITUDE_WEIGHTED: "Sparse factor column is mixed-sign and not a "
        "clean ±1 ternary; statistic is magnitude-weighted (Sefcik-Thompson) "
        "rather than textbook MacKinlay signed CAAR — apply .sign() before "
        "calling for sign-flip semantics.",
        WarningCode.FEW_EVENTS: "CAAR significance test with MIN_EVENTS_HARD ≤ "
        "n_event_periods < MIN_EVENTS_WARN (4..29). caar is an equal-weight "
        "calendar-time portfolio across event periods, so this counts the "
        "number of periods with an event, not events; a sub-30 series is "
        "power-thin for the asymptotic t-distribution — read borderline "
        "p-values cautiously.",
        WarningCode.BORDERLINE_PORTFOLIO_PERIODS: "top_concentration with MIN_PORTFOLIO_PERIODS_HARD "
        "≤ n_periods < MIN_PORTFOLIO_PERIODS_WARN (3..19); one-sided t-test "
        "on the per-date diversification ratio is returned but df=n-1 inflates "
        "t_crit relative to the asymptotic cutoff.",
        WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE: "Rectangular-kernel HAC variance-of-mean came out "
        "negative (no PSD guarantee, Andrews 1991); clamped to 0 → SE=0, t=0, p=1.0. "
        "Fires only on short / mildly anti-correlated samples.",
        WarningCode.UPSTREAM_UNAVAILABLE: "DAG-executor consumer skipped because an upstream "
        "producer short-circuited. The downstream MetricResult carries "
        "metadata['upstream'] / ['upstream_reason'] for the original cause.",
        WarningCode.CROSS_FACTOR_DENSITY_MISMATCH: "Factor columns carry inconsistent FactorDensity (dense and sparse mixed).",
        WarningCode.CROSS_FACTOR_SCOPE_MISMATCH: "Factor columns carry inconsistent FactorScope (individual and common mixed).",
        WarningCode.STRUCTURE_MISMATCH: "Metric's declared cell.structure disagrees with the data "
        "structure (e.g. a PANEL metric on TIMESERIES data); under "
        "strict=False the metric short-circuits to NaN instead of executing. "
        "metadata carries cell_structure / data_structure for diagnosis.",
        WarningCode.EXCESSIVE_PERIOD_DROPS: "An upstream PANEL→SERIES primitive dropped more than "
        "DROP_RATE_WARN_THRESHOLD of dates at its cross-sectional filter; the "
        "metric was computed on a shortened sample. Exact counts are in "
        "MetricResult.metadata (n_periods_in / n_periods_out / dropped_periods / "
        "drop_rate / drop_reason).",
        WarningCode.EXCESSIVE_ASSET_DROPS: "An upstream primitive dropped more than "
        "DROP_RATE_WARN_THRESHOLD of assets at its per-asset filter (e.g. "
        "compute_ts_betas dropping assets with insufficient history or zero "
        "factor variance); the cross-asset aggregate was computed on a shortened "
        "sample. Exact counts are in MetricResult.metadata (n_assets_in / "
        "n_assets_out / dropped_assets / drop_rate / drop_reason).",
    }
)


def cross_section_tier(n_assets: int) -> WarningCode | None:
    """Map an inference-stage cross-asset N to the appropriate warning code.

    The argument is the **inference-stage** N — the count of assets
    actually entering the cross-asset test, not the panel-union
    ``n_assets`` surface field. For ``(COMMON, *, None,
    PANEL)`` cells the two differ: ``compute_ts_betas`` drops assets
    with fewer than ``MIN_TS_PERIODS`` non-null observations, so the union
    can be materially larger than the post-filter count that drives
    ``primary_p``'s ``dof = N - 1``. Callers (``suggest_config``,
    ``_compute_common_panel``) therefore pre-filter before calling.

    A single :attr:`WarningCode.FEW_ASSETS` flags the whole thin regime
    (``2 ≤ n_assets < MIN_ASSETS_WARN``); how severe it is scales with
    ``n_assets``, which callers carry in metadata rather than encoding into
    separate tier members. Returns ``None`` at ``n_assets ≥ MIN_ASSETS_WARN``
    (clean) or ``n_assets < 2`` (PANEL impossible by upstream structure
    routing; defensive).
    """
    from factrix._stats.constants import MIN_ASSETS_WARN

    if 2 <= n_assets < MIN_ASSETS_WARN:
        return WarningCode.FEW_ASSETS
    return None
