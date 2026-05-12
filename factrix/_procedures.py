"""v0.5 ``FactorProcedure`` Protocol + per-cell procedures (§4.4.2 B3).

Each cell maps to exactly one ``FactorProcedure`` instance whose
``compute(raw, config) -> FactorProfile`` is the only place numerical
work happens.

Module-bottom ``register(...)`` calls populate
``factrix._registry._DISPATCH_REGISTRY`` at import time so the
registry SSOT (§4.4 A1) is queryable as soon as the package loads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, runtime_checkable

from factrix._axis import FactorScope, Metric, Mode, Signal
from factrix._codes import StatCode
from factrix._registry import _SCOPE_COLLAPSED, _DispatchKey, register

# ---------------------------------------------------------------------------
# Metadata helpers (#188)
# ---------------------------------------------------------------------------
# Hardcoded hyperparameters surfaced into ``profile.metadata`` so the
# StatCode they produced can be reproduced / audited without grepping
# the procedure source. Keep these constants here (not in
# ``_stats.constants``) so the metadata schema stays co-located with
# the code that emits it.
_ADF_LAG_ORDER = 0
_HHI_N_BINS = 10


def _panel_envelope(
    raw: Any,
    *,
    factor_col: str = "factor",
    return_col: str = "forward_return",
) -> tuple[int, int, int]:
    """Return ``(n_pairs, n_periods, n_assets)`` from the raw panel.

    ``n_pairs`` counts rows where both ``factor`` and ``forward_return``
    are non-null — the first-stage observation count the cell sees.
    ``n_periods`` / ``n_assets`` use the any-non-null union over the
    same two columns so callers reading ``n_assets = 1`` see the
    single-asset signal regardless of which column is sparser.
    """
    import polars as pl

    f_nn = pl.col(factor_col).is_not_null()
    r_nn = pl.col(return_col).is_not_null()
    any_nn = f_nn | r_nn
    row = raw.select(
        (f_nn & r_nn).sum().alias("n_pairs"),
        pl.col("date").filter(any_nn).n_unique().alias("n_periods"),
        pl.col("asset_id").filter(any_nn).n_unique().alias("n_assets"),
    ).row(0)
    return int(row[0]), int(row[1]), int(row[2])


def _nw_metadata(nw_lags: int) -> dict[StatCode, Mapping[str, Any]]:
    """NW HAC bandwidth → ``T_NW`` + ``P`` (shared single bandwidth choice).

    Returns a fresh inner dict per StatCode so downstream code that
    mutates one entry does not silently bleed into the other.
    """
    return {
        StatCode.T_NW: {"nw_lags": nw_lags},
        StatCode.P_NW: {"nw_lags": nw_lags},
    }


def _adf_metadata() -> dict[StatCode, Mapping[str, Any]]:
    return {
        StatCode.FACTOR_ADF_TAU: {"lag_order": _ADF_LAG_ORDER},
        StatCode.FACTOR_ADF_P: {"lag_order": _ADF_LAG_ORDER},
    }


def _ljung_box_metadata(lag_h: int) -> dict[StatCode, Mapping[str, Any]]:
    return {
        StatCode.RESID_LJUNG_BOX_Q: {"lag_h": lag_h},
        StatCode.RESID_LJUNG_BOX_P: {"lag_h": lag_h},
    }


def _hhi_metadata(n_bins: int) -> dict[StatCode, Mapping[str, Any]]:
    return {StatCode.EVENT_HHI_VALUE: {"n_bins": n_bins}}


def _hh_metadata(clamped: bool) -> dict[StatCode, Mapping[str, Any]]:
    """HH-pure rectangular-kernel HAC → ``T_HH`` / ``P_HH`` reproducibility record.

    ``variance_clamped=True`` mirrors the ``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE``
    flag (γ₀ + 2 Σγⱼ < 0 → SE clamped to 0 → t_HH = 0, p_HH = 1.0). The
    lag count ``h - 1`` is derivable from ``profile.config.forward_periods``
    so it is not duplicated here. An equivalent (but independent) record
    is stored under both ``T_HH`` and ``P_HH`` so downstream mutation of
    one entry does not silently bleed into the other — same contract
    ``_nw_metadata`` carries for the ``(T_NW, P)`` pair.
    """
    return {
        StatCode.T_HH: {"kernel": "rectangular", "variance_clamped": clamped},
        StatCode.P_HH: {"kernel": "rectangular", "variance_clamped": clamped},
    }


def _emit_hh_p(
    values: Any,
    *,
    forward_periods: int,
    stats: dict[StatCode, float],
    metadata: dict[StatCode, Mapping[str, Any]],
    warnings: frozenset[WarningCode],
) -> frozenset[WarningCode]:
    """Compute and stitch the HH t-test result into a procedure's outputs.

    No-op when ``forward_periods <= 1`` (no overlap → HH collapses to
    iid SE; ``HansenHodrick`` lands on the missing-stat error). Otherwise
    populates ``stats[P_HH]``, merges ``_hh_metadata``, and folds in
    ``RECT_KERNEL_NEGATIVE_VARIANCE`` when the rectangular-kernel sum
    came out negative. The series is consumed as-is (post-null-drop):
    same compromise NW makes — strictly the MA(h-1) structure assumes
    1-calendar-unit spacing, but null IC / FM dates are rare zero-variance
    cases and re-densifying would distort the mean.
    """
    from factrix._codes import WarningCode
    from factrix._stats import _hansen_hodrick_t_test

    if forward_periods <= 1:
        return warnings

    t_hh, p_hh, _, hh_clamped = _hansen_hodrick_t_test(
        values, forward_periods=forward_periods
    )
    stats[StatCode.T_HH] = t_hh
    stats[StatCode.P_HH] = p_hh
    metadata.update(_hh_metadata(hh_clamped))
    if hh_clamped:
        warnings |= frozenset({WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE})
    return warnings


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
    """Pure-compute contract: raw data + config → populated profile.

    ``EMITS_STATS`` is the **possible-set** of ``StatCode`` keys the
    procedure can populate on ``FactorProfile.stats`` (always-emitted ∪
    conditionally-emitted). ``describe_analysis_modes(format="json")``
    surfaces this so agents can cross-check that an ``Estimator``
    instance is dispatchable for the cell without running the
    procedure.
    """

    INPUT_SCHEMA: ClassVar[InputSchema]
    EMITS_STATS: ClassVar[frozenset[StatCode]]

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
    EMITS_STATS: ClassVar[frozenset[StatCode]] = frozenset(
        {
            StatCode.MEAN,
            StatCode.T_NW,
            StatCode.P_NW,
            # (T_HH, P_HH) pair, conditionally emitted when forward_periods > 1.
            StatCode.T_HH,
            StatCode.P_HH,
        }
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
        from factrix.metrics.ic import compute_ic

        # ``compute_ic`` filters by MIN_ASSETS_PER_DATE_IC but does not drop nulls;
        # ``pl.corr`` returns null for zero-variance dates (degenerate
        # factor / tied returns) so the explicit drop is reachable.
        ic_values = compute_ic(raw)["ic"].drop_nulls().to_numpy()
        n_periods = len(ic_values)
        n_pairs, n_periods_raw, n_assets = _panel_envelope(raw)
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

        stats: dict[StatCode, float] = {
            StatCode.MEAN: ic_mean,
            StatCode.T_NW: t_stat,
            StatCode.P_NW: p_value,
        }
        metadata = _nw_metadata(nw_lags)
        warnings = _emit_hh_p(
            ic_values,
            forward_periods=config.forward_periods,
            stats=stats,
            metadata=metadata,
            warnings=frozenset[WarningCode](),
        )

        return FactorProfile(
            config=config,
            mode=Mode.PANEL,
            primary_p=p_value,
            primary_stat=t_stat,
            primary_stat_name=StatCode.T_NW,
            n_obs=n_periods,
            n_pairs=n_pairs,
            n_periods=n_periods_raw,
            n_assets=n_assets,
            warnings=warnings,
            stats=stats,
            metadata=metadata,
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
    EMITS_STATS: ClassVar[frozenset[StatCode]] = frozenset(
        {
            StatCode.MEAN,
            StatCode.T_NW,
            StatCode.P_NW,
            # (T_HH, P_HH) pair, conditionally emitted when forward_periods > 1.
            StatCode.T_HH,
            StatCode.P_HH,
        }
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
        n_pairs, n_periods_raw, n_assets = _panel_envelope(raw)
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

        stats: dict[StatCode, float] = {
            StatCode.MEAN: lambda_mean,
            StatCode.T_NW: t_stat,
            StatCode.P_NW: p_value,
        }
        metadata = _nw_metadata(nw_lags)
        warning_codes = _emit_hh_p(
            beta_values,
            forward_periods=config.forward_periods,
            stats=stats,
            metadata=metadata,
            warnings=warning_codes,
        )

        return FactorProfile(
            config=config,
            mode=Mode.PANEL,
            primary_p=p_value,
            primary_stat=t_stat,
            primary_stat_name=StatCode.T_NW,
            n_obs=n_periods,
            n_pairs=n_pairs,
            n_periods=n_periods_raw,
            n_assets=n_assets,
            warnings=warning_codes,
            stats=stats,
            metadata=metadata,
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

    Output contract: ``MEAN`` reports the event-only mean
    (user-facing statistic — the average effect on event days);
    ``n_obs`` reflects the dense series the t-stat is computed on.
    ``mean_dense × n_total = mean_event × n_event``, so the t-statistic
    is invariant to the dense reframing in the iid limit (canonical p
    unchanged when the lag rule was already valid).
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )
    EMITS_STATS: ClassVar[frozenset[StatCode]] = frozenset(
        {
            StatCode.MEAN,
            StatCode.T_NW,
            StatCode.P_NW,
        }
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
        n_pairs, n_periods_raw, n_assets = _panel_envelope(raw)

        event_mean = (
            float(event_caar["caar"].drop_nulls().mean())  # type: ignore[arg-type]
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
            primary_stat=t_stat,
            primary_stat_name=StatCode.T_NW,
            n_obs=n_periods,
            n_pairs=n_pairs,
            n_periods=n_periods_raw,
            n_assets=n_assets,
            warnings=frozenset(warning_codes),
            stats={
                StatCode.MEAN: event_mean,
                StatCode.T_NW: t_stat,
                StatCode.P_NW: p_value,
            },
            metadata=_nw_metadata(nw_lags),
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
    EMITS_STATS: ClassVar[frozenset[StatCode]] = frozenset(
        {
            StatCode.MEAN,
            StatCode.T_NW,
            StatCode.P_NW,
            StatCode.FACTOR_ADF_TAU,
            StatCode.FACTOR_ADF_P,
        }
    )

    def compute(
        self,
        raw: Any,
        config: AnalysisConfig,
    ) -> FactorProfile:
        return _compute_common_panel(raw, config, with_adf=True)


class _CommonSparsePanelProcedure:
    """``(COMMON, SPARSE, None, PANEL)`` — broadcast event-dummy β panel.

    Per-asset OLS β_i on a broadcast sparse ``{0, R}`` dummy
    (canonical ``{-1, 0, +1}``), aggregated to a cross-asset t-test
    on ``E[β]``. Plan §4.3: same per-asset β →
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
    EMITS_STATS: ClassVar[frozenset[StatCode]] = frozenset(
        {
            StatCode.MEAN,
            StatCode.T_NW,
            StatCode.P_NW,
        }
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
                    "events are encoded with non-zero magnitude on event dates "
                    "(zero on non-event dates; canonical ±1) before dispatching"
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
        StatCode.MEAN: beta_mean,
        StatCode.T_NW: t_stat,
        StatCode.P_NW: p_value,
    }
    metadata: dict[StatCode, Mapping[str, Any]] = {}
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
        adf_tau, adf_p = _adf(factor_series, lags=_ADF_LAG_ORDER)
        stats[StatCode.FACTOR_ADF_TAU] = adf_tau
        stats[StatCode.FACTOR_ADF_P] = adf_p
        metadata.update(_adf_metadata())
        if adf_p > 0.10:
            warnings.add(WarningCode.PERSISTENT_REGRESSOR)

    # `n_obs == N` here (cross-asset aggregation) — distinct from the
    # `n_assets` envelope when the per-asset MIN_TS_OBS filter drops rows.
    n_pairs, n_periods_raw, n_assets = _panel_envelope(raw)
    return FactorProfile(
        config=config,
        mode=Mode.PANEL,
        primary_p=p_value,
        primary_stat=t_stat,
        primary_stat_name=StatCode.T_NW,
        n_obs=N,
        n_pairs=n_pairs,
        n_periods=n_periods_raw,
        n_assets=n_assets,
        warnings=frozenset(warnings),
        stats=stats,
        metadata=metadata,
    )


class _TSBetaContTimeseriesProcedure:
    """``(COMMON, CONTINUOUS, None, TIMESERIES)`` — single-asset OLS β.

    Plan §5.2 TIMESERIES continuous: OLS ``y_t = α + β·factor_t + ε`` with
    NW HAC SE on β; ADF on factor surfaces persistence (CONTINUOUS-only
    diagnostic per I6). n_periods-stratified per I5: below ``MIN_PERIODS_HARD`` raise
    ``InsufficientSampleError``; in ``[MIN_PERIODS_HARD, MIN_PERIODS_WARN)``
    emit result tagged with ``WarningCode.UNRELIABLE_SE_SHORT_PERIODS``.
    """

    INPUT_SCHEMA: ClassVar[InputSchema] = InputSchema(
        required_columns=("date", "asset_id", "factor", "forward_return"),
    )
    EMITS_STATS: ClassVar[frozenset[StatCode]] = frozenset(
        {
            StatCode.MEAN,
            StatCode.T_NW,
            StatCode.P_NW,
            StatCode.FACTOR_ADF_TAU,
            StatCode.FACTOR_ADF_P,
        }
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
        adf_tau, adf_p = _adf(x, lags=_ADF_LAG_ORDER)

        warnings: set[WarningCode] = set()
        if n_periods < MIN_PERIODS_WARN:
            warnings.add(WarningCode.UNRELIABLE_SE_SHORT_PERIODS)
        # I6: ADF persistence diagnostic is CONTINUOUS-only. The 0.10
        # cutoff matches plan §5.2 — a non-rejection at the 10% level
        # is the conventional "likely persistent" trigger.
        if adf_p > 0.10:
            warnings.add(WarningCode.PERSISTENT_REGRESSOR)

        n_pairs, n_periods_raw, n_assets = _panel_envelope(raw)
        return FactorProfile(
            config=config,
            mode=Mode.TIMESERIES,
            primary_p=p_value,
            primary_stat=t_stat,
            primary_stat_name=StatCode.T_NW,
            n_obs=n_periods,
            n_pairs=n_pairs,
            n_periods=n_periods_raw,
            n_assets=n_assets,
            warnings=frozenset(warnings),
            stats={
                StatCode.MEAN: beta,
                StatCode.T_NW: t_stat,
                StatCode.P_NW: p_value,
                StatCode.FACTOR_ADF_TAU: adf_tau,
                StatCode.FACTOR_ADF_P: adf_p,
            },
            metadata=_nw_metadata(nw_lags) | _adf_metadata(),
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
    EMITS_STATS: ClassVar[frozenset[StatCode]] = frozenset(
        {
            StatCode.MEAN,
            StatCode.T_NW,
            StatCode.P_NW,
            StatCode.RESID_LJUNG_BOX_Q,
            StatCode.RESID_LJUNG_BOX_P,
            StatCode.EVENT_HHI_VALUE,
        }
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
            _ljung_box,
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
        ljung_box_h, ljung_box_q, ljung_box_p = _ljung_box(resid)
        hhi = _event_temporal_hhi(d, n_bins=_HHI_N_BINS)
        overlap = _has_event_window_overlap(d, config.forward_periods)

        warnings: set[WarningCode] = set()
        if n_periods < MIN_PERIODS_WARN:
            warnings.add(WarningCode.UNRELIABLE_SE_SHORT_PERIODS)
        if ljung_box_p < 0.05:
            warnings.add(WarningCode.SERIAL_CORRELATION_DETECTED)
        if overlap:
            warnings.add(WarningCode.EVENT_WINDOW_OVERLAP)

        n_pairs, n_periods_raw, n_assets = _panel_envelope(raw)
        return FactorProfile(
            config=config,
            mode=Mode.TIMESERIES,
            primary_p=p_value,
            primary_stat=t_stat,
            primary_stat_name=StatCode.T_NW,
            n_obs=n_periods,
            n_pairs=n_pairs,
            n_periods=n_periods_raw,
            n_assets=n_assets,
            warnings=frozenset(warnings),
            stats={
                StatCode.MEAN: beta,
                StatCode.T_NW: t_stat,
                StatCode.P_NW: p_value,
                StatCode.RESID_LJUNG_BOX_Q: ljung_box_q,
                StatCode.RESID_LJUNG_BOX_P: ljung_box_p,
                StatCode.EVENT_HHI_VALUE: hhi,
            },
            metadata=(
                _nw_metadata(nw_lags)
                | _ljung_box_metadata(ljung_box_h)
                | _hhi_metadata(_HHI_N_BINS)
            ),
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
    evaluate_metric_name="ic",
)
register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.CONTINUOUS, Metric.FM, Mode.PANEL),
    _FMContPanelProcedure(),
    use_case="Fama-MacBeth λ on per-date OLS slope.",
    refs=("Fama & MacBeth (1973)", "Petersen (2009)"),
    evaluate_metric_name="fama_macbeth",
)
register(
    _DispatchKey(FactorScope.INDIVIDUAL, Signal.SPARSE, None, Mode.PANEL),
    _CAARSparsePanelProcedure(),
    use_case="Cross-event CAAR with t-test on per-event AR aggregate.",
    refs=("Brown & Warner (1985)", "MacKinlay (1997)"),
    evaluate_metric_name="caar",
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.PANEL),
    _CommonContPanelProcedure(),
    use_case="Per-asset β on broadcast factor + cross-asset t on E[β].",
    refs=("Black, Jensen & Scholes (1972)", "Fama & French (1993)"),
    evaluate_metric_name="ts_beta",
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.SPARSE, None, Mode.PANEL),
    _CommonSparsePanelProcedure(),
    use_case="Per-asset β on broadcast event dummy + cross-asset t.",
    refs=("MacKinlay (1997)",),
    evaluate_metric_name="ts_beta",
)
register(
    _DispatchKey(FactorScope.COMMON, Signal.CONTINUOUS, None, Mode.TIMESERIES),
    _TSBetaContTimeseriesProcedure(),
    use_case="Single-asset OLS β on broadcast factor + NW HAC SE.",
    refs=("Newey & West (1987, 1994)", "Stambaugh (1999)"),
    evaluate_metric_name="ts_beta",
)
register(
    _DispatchKey(_SCOPE_COLLAPSED, Signal.SPARSE, None, Mode.TIMESERIES),
    _TSDummySparseTimeseriesProcedure(),
    use_case="Single-asset calendar-time TS dummy regression + NW HAC SE.",
    refs=("Newey & West (1994)", "Ljung & Box (1978)"),
    evaluate_metric_name="ts_beta",
)
