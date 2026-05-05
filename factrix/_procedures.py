"""v0.5 ``FactorProcedure`` Protocol + per-cell procedures (§4.4.2 B3).

Each cell maps to exactly one ``FactorProcedure`` instance whose
``compute(raw, config) -> FactorProfile`` is the only place numerical
work happens.

Module-bottom ``register(...)`` calls populate
``factrix._registry._DISPATCH_REGISTRY`` at import time so the
registry SSOT (§4.4 A1) is queryable as soon as the package loads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._registry import _SCOPE_COLLAPSED, _DispatchKey, register

if TYPE_CHECKING:
    from factrix._analysis_config import AnalysisConfig
    from factrix._codes import WarningCode
    from factrix._profile import FactorProfile


@dataclass(frozen=True, slots=True)
class InputSchema:
    """Raw-data shape contract a procedure expects."""

    required_columns: tuple[str, ...] = ()


@runtime_checkable
class FactorProcedure(Protocol):
    """Pure-compute contract: raw data + config → populated profile."""

    INPUT_SCHEMA: ClassVar[InputSchema]

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile: ...


class _ICContPanelProcedure:
    """``(INDIVIDUAL, CONTINUOUS, IC, PANEL)`` — per-date Spearman IC.

    Aggregates per-date rank correlations between the factor and the
    forward-return into a time series, then runs an NW HAC t-test on
    its mean (Bartlett kernel, NW1994 automatic lag).
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        import numpy as np

        from factrix._codes import StatCode
        from factrix._profile import FactorProfile
        from factrix._stats import _newey_west_t_test, _resolve_nw_lags
        from factrix._stats.constants import auto_bartlett
        from factrix.metrics.ic import compute_ic

        # ``compute_ic`` filters by MIN_ASSETS_PER_DATE_IC but does not drop nulls;
        # ``pl.corr`` returns null for zero-variance dates (degenerate
        # factor / tied returns) so the explicit drop is reachable.
        ic_values = compute_ic(raw)["ic"].drop_nulls().to_numpy()
        n_periods = len(ic_values)
        n_assets = int(raw["asset_id"].n_unique())
        # Plan §5.2 picks NW1994 auto_bartlett as the default lag, but
        # h-period forward returns force MA(h-1) structure on the IC
        # series so we floor at ``forward_periods - 1`` (Hansen-Hodrick
        # 1980) to keep the HAC SE consistent. ``_resolve_nw_lags``
        # applies that floor and the ``min(., n_periods-1)`` clip in one place.
        nw_lags = (
            _resolve_nw_lags(
                n_periods, auto_bartlett(n_periods), config.forward_periods
            )
            if n_periods >= 2
            else 0
        )
        ic_mean = float(np.mean(ic_values)) if n_periods > 0 else 0.0
        t_stat, p_value, _ = _newey_west_t_test(ic_values, lags=nw_lags)

        return FactorProfile(
            config=config,
            mode=Mode.PANEL,
            primary_p=p_value,
            n_obs=n_periods,
            n_assets=n_assets,
            stats={
                StatCode.IC_MEAN: ic_mean,
                StatCode.IC_T_NW: t_stat,
                StatCode.IC_P: p_value,
                StatCode.NW_LAGS_USED: float(nw_lags),
            },
        )


class _FMContPanelProcedure:
    """``(INDIVIDUAL, CONTINUOUS, FM, PANEL)`` — per-date OLS slope λ.

    Aggregates per-date cross-sectional regression slopes into a time
    series, then runs an NW HAC t-test on the mean (Bartlett kernel,
    NW1994 auto lag with Hansen-Hodrick overlap floor — same shape as
    the IC procedure).
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        import numpy as np

        from factrix._codes import StatCode, WarningCode
        from factrix._profile import FactorProfile
        from factrix._stats import _newey_west_t_test, _resolve_nw_lags
        from factrix._stats.constants import auto_bartlett
        from factrix.metrics.fama_macbeth import (
            MIN_FM_PERIODS_HARD,
            MIN_FM_PERIODS_WARN,
            compute_fm_betas,
        )

        beta_values = compute_fm_betas(raw)["beta"].drop_nulls().to_numpy()
        n_periods = len(beta_values)
        n_assets = int(raw["asset_id"].n_unique())
        nw_lags = (
            _resolve_nw_lags(
                n_periods, auto_bartlett(n_periods), config.forward_periods
            )
            if n_periods >= 2
            else 0
        )
        lambda_mean = float(np.mean(beta_values)) if n_periods > 0 else 0.0
        t_stat, p_value, _ = _newey_west_t_test(beta_values, lags=nw_lags)

        warning_codes: frozenset[WarningCode] = (
            frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})
            if MIN_FM_PERIODS_HARD <= n_periods < MIN_FM_PERIODS_WARN
            else frozenset()
        )

        return FactorProfile(
            config=config,
            mode=Mode.PANEL,
            primary_p=p_value,
            n_obs=n_periods,
            n_assets=n_assets,
            warnings=warning_codes,
            stats={
                StatCode.FM_LAMBDA_MEAN: lambda_mean,
                StatCode.FM_LAMBDA_T_NW: t_stat,
                StatCode.FM_LAMBDA_P: p_value,
                StatCode.NW_LAGS_USED: float(nw_lags),
            },
        )


class _CAARSparsePanelProcedure:
    """``(INDIVIDUAL, SPARSE, None, PANEL)`` — calendar-time CAAR.

    Plan §4.3 / Issue #24. Dispatches to ``compute_caar`` (see for the
    magnitude-preserving per-row formula), reindexes the event-date-
    indexed series to the **dense calendar** (zero-fill on non-event
    dates), then runs an HH-floored NW HAC t-test on the dense series.

    Densification is the calendar-time portfolio approach (Jaffe 1974,
    Mandelker 1974; Fama 1998 §2) — restores the lag rule's "consecutive
    observations are 1 calendar period apart" assumption that
    ``compute_caar``'s event-date filter would otherwise break. With it,
    sparse events let zero-padding zero out spurious autocovariance
    terms and clustered events get the real MA(h-1) overlap structure
    weighted correctly. Pipeline parity with IC / FM / common-sparse
    PANEL: same ``_resolve_nw_lags`` + ``_newey_west_t_test`` machinery,
    same dense-series semantics.

    Output contract: ``CAAR_MEAN`` reports the event-only mean
    (user-facing statistic — the average effect on event days);
    ``n_obs`` and ``NW_LAGS_USED`` reflect the dense series the t-stat
    is computed on. ``mean_dense × n_total = mean_event × n_event``, so
    the t-statistic is invariant to the dense reframing in the iid
    limit (canonical p unchanged when the lag rule was already valid).
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        import polars as pl

        from factrix._codes import StatCode, WarningCode
        from factrix._profile import FactorProfile
        from factrix._stats import _newey_west_t_test, _resolve_nw_lags
        from factrix._stats.constants import auto_bartlett
        from factrix._types import MIN_EVENTS_HARD, MIN_EVENTS_WARN
        from factrix.metrics._helpers import _is_sparse_magnitude_weighted
        from factrix.metrics.caar import compute_caar

        warning_codes: set[WarningCode] = set()
        if _is_sparse_magnitude_weighted(raw, "factor"):
            warning_codes.add(WarningCode.SPARSE_MAGNITUDE_WEIGHTED)

        event_caar = compute_caar(raw)
        n_event_dates = event_caar.height
        if MIN_EVENTS_HARD <= n_event_dates < MIN_EVENTS_WARN:
            warning_codes.add(WarningCode.FEW_EVENTS_BROWN_WARNER)

        all_dates = raw.select(pl.col("date").unique().sort())
        dense = (
            all_dates.join(event_caar, on="date", how="left")
            .with_columns(pl.col("caar").fill_null(0.0))
            .sort("date")
        )
        caar_dense = dense["caar"].to_numpy()
        n_periods = len(caar_dense)
        n_assets = int(raw["asset_id"].n_unique())

        event_mean = (
            float(event_caar["caar"].drop_nulls().mean())
            if event_caar.height > 0
            else 0.0
        )
        nw_lags = (
            _resolve_nw_lags(
                n_periods,
                auto_bartlett(n_periods),
                config.forward_periods,
            )
            if n_periods >= 2
            else 0
        )
        t_stat, p_value, _ = _newey_west_t_test(caar_dense, lags=nw_lags)

        return FactorProfile(
            config=config,
            mode=Mode.PANEL,
            primary_p=p_value,
            n_obs=n_periods,
            n_assets=n_assets,
            warnings=frozenset(warning_codes),
            stats={
                StatCode.CAAR_MEAN: event_mean,
                StatCode.CAAR_T_NW: t_stat,
                StatCode.CAAR_P: p_value,
                StatCode.NW_LAGS_USED: float(nw_lags),
            },
        )


class _CommonContPanelProcedure:
    """``(COMMON, CONTINUOUS, None, PANEL)`` — broadcast factor β panel.

    Per-asset OLS β_i on the broadcast factor (single time series shared
    across assets), aggregated to a cross-asset t-test on ``E[β]``.
    ADF on the factor surfaces persistence (CONTINUOUS-only per I6).
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        return _compute_common_panel(raw, config, with_adf=True)


class _CommonSparsePanelProcedure:
    """``(COMMON, SPARSE, None, PANEL)`` — broadcast event-dummy β panel.

    Per-asset OLS β_i on a broadcast {-1, 0, +1} dummy, aggregated to
    a cross-asset t-test on ``E[β]``. Plan §4.3: same per-asset β →
    cross-asset t-test pattern as COMMON × CONTINUOUS, with the dummy
    replacing the continuous factor. ADF skipped (I6).

    Two-tier event-count guard on the broadcast dummy: with very few
    non-zero events the per-asset β is fit from a handful of points,
    yielding an asymptotic t the cross-asset aggregation cannot
    rescue. Below ``MIN_BROADCAST_EVENTS_HARD`` raises; the borderline
    tier surfaces ``SPARSE_COMMON_FEW_EVENTS``.
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        import polars as pl

        from factrix._codes import WarningCode
        from factrix._errors import InsufficientSampleError
        from factrix._stats.constants import (
            MIN_BROADCAST_EVENTS_HARD,
            MIN_BROADCAST_EVENTS_WARN,
        )
        from factrix.metrics._helpers import _is_sparse_magnitude_weighted

        # Broadcast factor is the same per date across assets; count
        # event dates by collapsing to one row per date first.
        n_events = int(
            raw.group_by("date")
            .agg(pl.col("factor").first())
            .filter(pl.col("factor") != 0)
            .height,
        )
        if n_events < MIN_BROADCAST_EVENTS_HARD:
            if n_events == 0:
                detail = (
                    "the broadcast factor has no non-zero observations; ensure "
                    "events are encoded as ±1 on event dates (zero on non-event "
                    "dates) before dispatching"
                )
            else:
                detail = (
                    "aggregate to a coarser frequency, broaden the event "
                    "definition, or switch to a continuous factory"
                )
            raise InsufficientSampleError(
                f"n_events={n_events} below "
                f"MIN_BROADCAST_EVENTS_HARD={MIN_BROADCAST_EVENTS_HARD}; "
                f"per-asset β on a broadcast sparse dummy is fit from too few "
                f"informative observations to support the cross-asset t-test. "
                f"To resolve: {detail}.",
                actual_periods=n_events,
                required_periods=MIN_BROADCAST_EVENTS_HARD,
            )

        extra_codes: set[WarningCode] = set()
        if n_events < MIN_BROADCAST_EVENTS_WARN:
            extra_codes.add(WarningCode.SPARSE_COMMON_FEW_EVENTS)
        if _is_sparse_magnitude_weighted(raw, "factor"):
            extra_codes.add(WarningCode.SPARSE_MAGNITUDE_WEIGHTED)
        extra_warnings: frozenset[WarningCode] = frozenset(extra_codes)
        return _compute_common_panel(
            raw,
            config,
            with_adf=False,
            extra_warnings=extra_warnings,
        )


def _compute_common_panel(
    raw: Any,
    config: AnalysisConfig,
    *,
    with_adf: bool,
    extra_warnings: frozenset[WarningCode] = frozenset(),
) -> FactorProfile:
    """Shared core for the two ``(COMMON, *, None, PANEL)`` procedures.

    The two cells differ only in (a) whether ADF persistence is run on
    the broadcast factor and (b) what the regressor's marginal looks
    like; ``compute_ts_betas`` treats both identically (per-asset OLS).

    The cross-asset SE assumes asset-level independence (plan §4.3 spec).
    Under contemporaneous return correlation across assets — common in
    market-driven panels — the standard t will over-state significance;
    Petersen (2009) clustered SE is deferred per plan §11.
    """
    import numpy as np
    import polars as pl

    from factrix._codes import StatCode, WarningCode, cross_section_tier
    from factrix._profile import FactorProfile
    from factrix._stats import _adf, _calc_t_stat, _p_value_from_t
    from factrix.metrics.ts_beta import compute_ts_betas

    betas_df = compute_ts_betas(raw)
    betas = betas_df["beta"].drop_nulls().to_numpy()
    N = len(betas)

    if N == 0:
        beta_mean = 0.0
        t_stat = 0.0
        p_value = 1.0
    else:
        beta_mean = float(np.mean(betas))
        beta_std = float(np.std(betas, ddof=1)) if N >= 2 else 0.0
        t_stat = _calc_t_stat(beta_mean, beta_std, N)
        p_value = _p_value_from_t(t_stat, N) if N >= 2 else 1.0

    stats: dict[StatCode, float] = {
        StatCode.TS_BETA: beta_mean,
        StatCode.TS_BETA_T_NW: t_stat,
        StatCode.TS_BETA_P: p_value,
    }
    warnings: set[WarningCode] = set(extra_warnings)
    n_tier = cross_section_tier(N)
    if n_tier is not None:
        warnings.add(n_tier)

    if with_adf:
        # The broadcast factor is the same series across every asset on
        # a given date; collapse to one row per date before ADF.
        factor_series = (
            raw.sort("date")
            .group_by("date", maintain_order=True)
            .agg(pl.col("factor").first())["factor"]
            .drop_nulls()
            .to_numpy()
        )
        _, adf_p = _adf(factor_series)
        stats[StatCode.FACTOR_ADF_P] = adf_p
        if adf_p > 0.10:
            warnings.add(WarningCode.PERSISTENT_REGRESSOR)

    # n_obs == N here (cross-asset aggregation), but expose n_assets
    # explicitly anyway so callers get the same field shape regardless
    # of which cell produced the profile.
    return FactorProfile(
        config=config,
        mode=Mode.PANEL,
        primary_p=p_value,
        n_obs=N,
        n_assets=int(raw["asset_id"].n_unique()),
        warnings=frozenset(warnings),
        stats=stats,
    )


class _TSBetaContTimeseriesProcedure:
    """``(COMMON, CONTINUOUS, None, TIMESERIES)`` — single-asset OLS β.

    Plan §5.2 TIMESERIES continuous: OLS ``y_t = α + β·factor_t + ε`` with
    NW HAC SE on β; ADF on factor surfaces persistence (CONTINUOUS-only
    diagnostic per I6). n_periods-stratified per I5: below ``MIN_PERIODS_HARD`` raise
    ``InsufficientSampleError``; in ``[MIN_PERIODS_HARD, MIN_PERIODS_WARN)``
    emit verdict + ``WarningCode.UNRELIABLE_SE_SHORT_PERIODS``.
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        from factrix._codes import StatCode, WarningCode
        from factrix._errors import InsufficientSampleError
        from factrix._profile import FactorProfile
        from factrix._stats import _adf, _ols_nw_slope_t, _resolve_nw_lags
        from factrix._stats.constants import (
            MIN_PERIODS_HARD,
            MIN_PERIODS_WARN,
            auto_bartlett,
        )

        sorted_raw = raw.sort("date")
        y = sorted_raw["forward_return"].drop_nulls().to_numpy()
        x = sorted_raw["factor"].drop_nulls().to_numpy()
        n_periods = int(min(len(y), len(x)))

        if n_periods < MIN_PERIODS_HARD:
            raise InsufficientSampleError(
                f"n_periods={n_periods} below MIN_PERIODS_HARD={MIN_PERIODS_HARD}; NW HAC SE is too "
                "biased for primary_p to be trustworthy at this floor. "
                f"Extend the time series to ≥{MIN_PERIODS_HARD} rows or "
                "aggregate to a lower frequency.",
                actual_periods=n_periods,
                required_periods=MIN_PERIODS_HARD,
            )

        # Same HH overlap floor logic as the IC PANEL procedure: forward
        # returns of horizon h induce MA(h-1) structure that the auto
        # Bartlett rule does not see on its own.
        nw_lags = _resolve_nw_lags(
            n_periods,
            auto_bartlett(n_periods),
            config.forward_periods,
        )
        # Truncate to common length on the off-chance one column had
        # extra nulls.
        y, x = y[:n_periods], x[:n_periods]
        beta, t_stat, p_value, _ = _ols_nw_slope_t(y, x, lags=nw_lags)
        _adf_tau, adf_p = _adf(x)

        warnings: set[WarningCode] = set()
        if n_periods < MIN_PERIODS_WARN:
            warnings.add(WarningCode.UNRELIABLE_SE_SHORT_PERIODS)
        # I6: ADF persistence diagnostic is CONTINUOUS-only. The 0.10
        # cutoff matches plan §5.2 — a non-rejection at the 10% level
        # is the conventional "likely persistent" trigger.
        if adf_p > 0.10:
            warnings.add(WarningCode.PERSISTENT_REGRESSOR)

        return FactorProfile(
            config=config,
            mode=Mode.TIMESERIES,
            primary_p=p_value,
            n_obs=n_periods,
            n_assets=int(raw["asset_id"].n_unique()),
            warnings=frozenset(warnings),
            stats={
                StatCode.TS_BETA: beta,
                StatCode.TS_BETA_T_NW: t_stat,
                StatCode.TS_BETA_P: p_value,
                StatCode.FACTOR_ADF_P: adf_p,
                StatCode.NW_LAGS_USED: float(nw_lags),
            },
        )


class _TSDummySparseTimeseriesProcedure:
    """``(_, SPARSE, None, TIMESERIES)`` — single-asset OLS β on dummy.

    Plan §5.2 TIMESERIES sparse / §5.4.1 sentinel collapse: shared across
    both user-facing scopes (``individual_sparse`` and ``common_sparse``)
    because at N=1 they are statistically equivalent. ``primary_p``
    is the calendar-time TS dummy regression β NW HAC t-test — NOT
    event-time CAAR (CAAR is kept as a PANEL procedure for the
    PANEL cell).

    Diagnostics (plan §5.2 four-layer warnings):

    1. event window overlap on consecutive event gaps
    2. Ljung-Box on residual ε_t (auto-lag ``min(10, n_periods//10)``)
    3. ``event_temporal_hhi`` Herfindahl on equal-time bin shares —
       surfaces clustering of events along the calendar axis
    4. ``UNRELIABLE_SE_SHORT_PERIODS`` for ``n_periods < MIN_PERIODS_WARN``
       (n_periods < ``MIN_PERIODS_HARD`` raises ``InsufficientSampleError`` upstream)
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        from factrix._codes import StatCode, WarningCode
        from factrix._errors import InsufficientSampleError
        from factrix._profile import FactorProfile
        from factrix._stats import (
            _ljung_box_p,
            _ols_nw_slope_t,
            _resolve_nw_lags,
        )
        from factrix._stats.constants import (
            MIN_PERIODS_HARD,
            MIN_PERIODS_WARN,
            auto_bartlett,
        )

        sorted_raw = raw.sort("date")
        y = sorted_raw["forward_return"].drop_nulls().to_numpy()
        d = sorted_raw["factor"].drop_nulls().to_numpy()
        n_periods = int(min(len(y), len(d)))

        if n_periods < MIN_PERIODS_HARD:
            raise InsufficientSampleError(
                f"n_periods={n_periods} below MIN_PERIODS_HARD={MIN_PERIODS_HARD}; NW HAC SE is too "
                "biased for primary_p to be trustworthy at this floor. "
                f"Extend the time series to ≥{MIN_PERIODS_HARD} rows or "
                "aggregate to a lower frequency.",
                actual_periods=n_periods,
                required_periods=MIN_PERIODS_HARD,
            )

        y, d = y[:n_periods], d[:n_periods]
        nw_lags = _resolve_nw_lags(
            n_periods,
            auto_bartlett(n_periods),
            config.forward_periods,
        )
        beta, t_stat, p_value, resid = _ols_nw_slope_t(y, d, lags=nw_lags)
        ljung_box_p = _ljung_box_p(resid)
        hhi = _event_temporal_hhi(d)
        overlap = _has_event_window_overlap(d, config.forward_periods)

        warnings: set[WarningCode] = set()
        if n_periods < MIN_PERIODS_WARN:
            warnings.add(WarningCode.UNRELIABLE_SE_SHORT_PERIODS)
        if ljung_box_p < 0.05:
            warnings.add(WarningCode.SERIAL_CORRELATION_DETECTED)
        if overlap:
            warnings.add(WarningCode.EVENT_WINDOW_OVERLAP)

        return FactorProfile(
            config=config,
            mode=Mode.TIMESERIES,
            primary_p=p_value,
            n_obs=n_periods,
            n_assets=int(raw["asset_id"].n_unique()),
            warnings=frozenset(warnings),
            stats={
                StatCode.TS_BETA: beta,
                StatCode.TS_BETA_T_NW: t_stat,
                StatCode.TS_BETA_P: p_value,
                StatCode.LJUNG_BOX_P: ljung_box_p,
                StatCode.EVENT_TEMPORAL_HHI: hhi,
                StatCode.NW_LAGS_USED: float(nw_lags),
            },
        )


def _event_temporal_hhi(d_signal: Any, *, n_bins: int = 10) -> float:
    """Herfindahl share of events across ``n_bins`` equal-time calendar bins.

    HHI = ``Σ_k (n_k / N_total)²``. ``1/n_bins`` under uniform spread,
    ``1.0`` when all events land in one bin, ``0.0`` for an empty
    signal. Captures temporal clustering distinct from inter-event
    gap variance.
    """
    import numpy as np

    arr = np.asarray(d_signal)
    nonzero = arr != 0
    n_events = int(nonzero.sum())
    if n_events == 0:
        return 0.0
    n_periods = len(arr)
    bins_for_position = np.minimum(
        np.arange(n_periods) * n_bins // n_periods, n_bins - 1
    )
    counts = np.bincount(bins_for_position[nonzero], minlength=n_bins)
    shares = counts / n_events
    return float(np.sum(shares * shares))


def _has_event_window_overlap(d_signal: Any, forward_periods: int) -> bool:
    """True when any consecutive event pair sits within ``2*forward_periods``.

    Plan §5.2: the response window of one event still contaminates the
    next when ``min(dt_between_events) < 2 * window_length``.
    """
    import numpy as np

    arr = np.asarray(d_signal)
    positions = np.flatnonzero(arr != 0)
    if len(positions) < 2:
        return False
    return bool(np.min(np.diff(positions)) < 2 * forward_periods)


# ---------------------------------------------------------------------------
# Registry population
# ---------------------------------------------------------------------------
# (INDIVIDUAL, CONTINUOUS, *, TIMESERIES) intentionally absent —
# evaluate() raises ModeAxisError there (§5.5).

register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.IC, Mode.PANEL),
    _ICContPanelProcedure(),
    use_case="Per-date Spearman IC across the asset cross-section.",
    refs=("Grinold (1989)", "Newey & West (1987)"),
)
register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.FM, Mode.PANEL),
    _FMContPanelProcedure(),
    use_case="Fama-MacBeth λ on per-date OLS slope.",
    refs=("Fama & MacBeth (1973)", "Petersen (2009)"),
)
register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.SPARSE, None, Mode.PANEL),
    _CAARSparsePanelProcedure(),
    use_case="Cross-event CAAR with t-test on per-event AR aggregate.",
    refs=("Brown & Warner (1985)", "MacKinlay (1997)"),
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.PANEL),
    _CommonContPanelProcedure(),
    use_case="Per-asset β on broadcast factor + cross-asset t on E[β].",
    refs=("Black, Jensen & Scholes (1972)", "Fama & French (1993)"),
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.SPARSE, None, Mode.PANEL),
    _CommonSparsePanelProcedure(),
    use_case="Per-asset β on broadcast event dummy + cross-asset t.",
    refs=("MacKinlay (1997)",),
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.TIMESERIES),
    _TSBetaContTimeseriesProcedure(),
    use_case="Single-asset OLS β on broadcast factor + NW HAC SE.",
    refs=("Newey & West (1987, 1994)", "Stambaugh (1999)"),
)
register(
    _DispatchKey(_SCOPE_COLLAPSED, Signal.SPARSE, None, Mode.TIMESERIES),
    _TSDummySparseTimeseriesProcedure(),
    use_case="Single-asset calendar-time TS dummy regression + NW HAC SE.",
    refs=("Newey & West (1994)", "Ljung & Box (1978)"),
)
