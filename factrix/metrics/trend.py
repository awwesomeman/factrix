"""Information coefficient (IC) trend analysis using Theil-Sen estimator.

Theil-Sen is preferred over ordinary least squares (OLS) because it has a breakdown point of 29.3%,
making it robust to outliers (e.g. COVID-era IC spikes).

Notes:
    **Pipeline.** Time-series only, Theil-Sen median pairwise slope on
    a 1-D series; CI from the rank-based pairwise slope distribution.

    **Input.** DataFrame with ``date, value`` (typically an IC series).

    **Output.** Slope + confidence interval for trend detection.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from scipy import stats as sp_stats

from factrix._axis import (
    Aggregation,
    DataStructure,
    FactorDensity,
    FactorScope,
    InputShape,
)
from factrix._codes import WarningCode
from factrix._metric_index import cell
from factrix._results import MetricResult
from factrix._stats import _adf, _mann_kendall_hamed_rao
from factrix._stats.constants import PERSISTENT_SERIES_AUTOCORR
from factrix._stats.diagnostics import _lag1_autocorr
from factrix.metrics._decorators import metric
from factrix.metrics._helpers import (
    _degenerate_test_fields,
    _enforce_scaled_floor,
    _resolve_series_value_col,
    _sample_non_overlapping,
    _scaled_periods_threshold,
    _short_circuit_output,
    _surface_null_drop,
)
from factrix.metrics.ic import compute_ic

__all__ = [
    "ic_trend",
]

#: Minimum *non-overlapping* observations the trend test needs. The raw
#: series must supply ``_MIN_TREND_PERIODS * overlap_periods`` periods.
_MIN_TREND_PERIODS = 10


@metric(
    cell=cell(
        FactorScope.INDIVIDUAL, FactorDensity.DENSE, structure=DataStructure.PANEL
    ),
    aggregation=Aggregation.TS_ONLY,
    slice_boundary_sensitive=True,
    input_shape=InputShape.SERIES,
    requires={"series": compute_ic},
    # The trend test runs on the non-overlapping subsample, so the periods
    # floor scales with the stride exactly as ``positive_rate``'s does.
    sample_threshold=_scaled_periods_threshold(_MIN_TREND_PERIODS),
)
def ic_trend(
    series: pl.DataFrame,
    value_col: str = "value",
    overlap_periods: int = 5,
    *,
    name: str = "ic_trend",
    adf_threshold: float | None = 0.10,
    expected_warnings: tuple[str, ...] = (),
) -> MetricResult:
    """Theil-Sen median slope of a time-indexed series.

    Answers "is this factor getting better or worse over time?"
    - slope ≈ 0: stable
    - slope significantly < 0: decaying (crowding / alpha erosion)
    - slope significantly > 0: improving

    Args:
        series: DataFrame with ``date`` and ``value_col``.
        overlap_periods: Overlap horizon of the input series, used as the
            non-overlapping sampling stride. The default input is a
            per-period IC series built from h-period forward returns, so
            consecutive observations share ``h - 1`` bars; the trend test
            runs on every ``h``-th observation. Pass ``1`` only when the
            series genuinely carries no overlap.
        name: Emitted metric name, stashed in ``metadata`` and used as
            the method / cache key. Defaults to ``"ic_trend"``;
            EventFactor.caar_trend / MacroPanelFactor.beta_trend pass
            their own names so method / cache key / primitive name stay
            three-point unified.
        adf_threshold: Augmented Dickey-Fuller (ADF) p-value above which the input is flagged as
            unit-root suspect. Default ``0.10`` is a conventional
            practitioner cutoff from the unit-root literature (folklore
            on the back of [Stock-Watson (1988)][stock-watson-1988]'s
            broader review of trends in macroeconomic time series, with
            the specific 10% threshold closer to Stock 1994 *Handbook of
            Econometrics* §III than a direct prescription of the 1988
            paper): at p > 0.10 we cannot reject I(1), so ordinary least
            squares (OLS) / Theil-Sen on the series reject the slope
            null at inflated rates regardless of the true trend. When
            ``None``, the ADF check is skipped entirely and no
            ``adf_stat`` / ``adf_p`` / ``unit_root_suspected`` keys are
            written. When a float is provided it must lie in (0, 1).
            Detected unit roots set ``unit_root_suspected=True`` in
            metadata; the slope value is still returned (caller decides)
            but significance should be read with scepticism.

    Returns:
        MetricResult with value = slope, t_stat from Theil-Sen confidence interval.

    Notes:
        Theil-Sen median pairwise slope: ``slope = median{(y_j - y_i) /
        (j - i) : i < j}`` ([Sen (1968)][sen-1968]) reports the trend's
        *magnitude*; significance comes from the Mann-Kendall test
        ([Mann (1945)][mann-1945], [Kendall (1975)][kendall-1975]) on the
        same ranks — ``stat`` is Kendall's ``tau`` between the sequence
        index and the series. That is the standard pairing for a
        non-parametric trend test: one rank-based framework supplies both
        numbers. An ADF unit-root pre-check on the input flags series for
        which the trend null is rejected at inflated rates regardless of
        the true trend, and now carries
        ``WarningCode.PERSISTENT_REGRESSOR`` rather than sitting in
        metadata alone.

        **Two corrections for serial dependence.** Striding does ~99% of
        the work; Hamed-Rao handles the residual persistence that
        striding cannot reach, and is not calibrated on its own.
        Mann-Kendall's null variance
        ``Var(S) = n(n-1)(2n+5)/18`` counts every pairwise comparison as
        an independent draw. The default input to this metric — a
        per-period IC series over h-period forward returns — is MA(h-1)
        with ``rho_1 = 1 - 1/h`` (0.80 at h=5, 0.95 at h=21), so the raw
        test is wholesale miscalibrated and does not improve with ``T``.
        So:

        1. The series is sub-sampled to non-overlapping observations at
           stride ``overlap_periods`` (the same ``_sample_non_overlapping``
           machinery ``positive_rate`` uses) before anything is computed.
           This is exactly calibrated for the overlap channel.
        2. The Mann-Kendall variance on the survivors carries the
           [Hamed-Rao (1998)][hamed-rao-1998] ``n/n*`` inflation for the
           *residual* persistence striding cannot remove (an AR factor is
           still autocorrelated after striding). Reported as
           ``metadata["variance_inflation"]``.

        Measured rejection frequency on a true null (slope 0), 600
        replications per cell:

        | series             | raw MK | Hamed-Rao only | strided + HR |
        |--------------------|--------|----------------|--------------|
        | T=60,  h=5         | 0.405  | 0.215          | 0.045        |
        | T=240, h=5         | 0.367  | 0.162          | 0.038        |
        | T=120, h=21        | 0.683  | 0.312          | 0.022        |
        | T=240, h=21        | 0.673  | 0.265          | 0.033        |
        | T=120, iid         | 0.035  | 0.032          | 0.032        |
        | T=120, AR(0.6)     | 0.338  | 0.183          | 0.183        |
        | T=120, AR(0.9)     | 0.667  | 0.357          | 0.357        |

        The last two rows are the disclosed limit: striding does nothing
        for a persistent series at ``h = 1``, and Hamed-Rao only halves
        the excess. Read the other way round, in every *overlap* cell the
        strided and strided+Hamed-Rao columns agree to within 1pp — the
        stride is the fix, and Hamed-Rao earns its place only on the
        ``h = 1`` autocorrelated rows. Those runs carry
        ``WarningCode.SERIAL_CORRELATION_DETECTED``, fired off the lag-1
        autocorrelation of the strided series against
        ``PERSISTENT_SERIES_AUTOCORR``.

        **The size fix costs power, and at long horizons costs most of
        it.** Against a drift of 2 sd over the sample, raw Mann-Kendall
        versus strided + Hamed-Rao:

        | cell        | raw MK | strided + HR |
        |-------------|--------|--------------|
        | T=60,  h=5  | 0.865  | 0.360        |
        | T=120, h=5  | 0.971  | 0.713        |
        | T=240, h=5  | 0.999  | 0.952        |
        | T=120, h=21 | 0.888  | 0.086        |
        | T=240, h=21 | 0.951  | 0.425        |

        The raw column is not a fair comparison — it is the power of a
        test that rejects 37–68% under the null — but the absolute level
        of the corrected column is the number to plan around. At
        ``h = 21`` the stride leaves 12 survivors at ``T = 120`` and the
        metric is effectively powerless there; it needs roughly 240+
        periods before it can detect anything, and more still to be
        comparable with the ``h = 5`` cells.

        **Why not a t backed out of the CI.** An earlier version derived
        ``SE ≈ (ci_high - ci_low) / 2 / 1.96`` from scipy's normal-based
        95% rank interval and read ``slope / SE`` against ``t(n - 2)``.
        Under a true null that combination rejected 2.4–3.8% at a nominal
        5% for ``n = 10–30`` — conservative, but by mixing a normal-derived
        SE with a t reference rather than by design, and it made a
        perfectly linear series (zero-width CI) a special case that had to
        be hand-mapped to ``p = 0``. Mann-Kendall sits at 4.6–5.4% on the
        same nulls and handles the perfect line naturally (``tau = ±1``,
        exact p).

        **Constant series.** A flat series has no rank ordering to test:
        ``tau`` is undefined, so ``stat`` / ``p_value`` are ``None`` with
        ``WarningCode.DEGENERATE_VARIANCE``, while ``value`` (the zero
        slope) is still reported — the same convention every other metric
        uses for a dispersion-free sample. The Theil-Sen CI is kept in
        ``metadata`` as a descriptive interval; ``ci_excludes_zero`` is
        informational and no longer drives the p.

        factrix uses Theil-Sen rather than OLS because its 29.3% breakdown
        point absorbs information coefficient (IC) outliers (e.g. COVID-era spikes) that would
        dominate an OLS slope; the trade-off is the SE recovered from the
        rank-CI is approximate, not asymptotically exact.

    References:
        [Sen 1968][sen-1968]: Theil-Sen median pairwise slope.
        [Lou-Polk 2022][lou-polk-2022]: momentum-crowding evidence
        as one suggestive economic channel for time-varying IC;
        [McLean-Pontiff 2016][mclean-pontiff-2016] is the cleaner
        cite for post-publication IC decay specifically.
        [Hamed-Rao 1998][hamed-rao-1998]: variance-inflation
        correction making Mann-Kendall valid under serial correlation.
        [Stock-Watson 1988][stock-watson-1988]: practitioner
        unit-root background for the ADF persistence flag.
        [Dickey-Fuller 1979][dickey-fuller-1979]: ADF persistence
        diagnostic on the input series.
        [MacKinnon 1996][mackinnon-1996]: ADF p-value response surface
        used by ``_adf_pvalue_interp``.

    Examples:
        Trend on the per-period IC series produced by
        :func:`~factrix.metrics.ic.compute_ic`:

        >>> import factrix as fx
        >>> from factrix.preprocess import compute_forward_return
        >>> from factrix.metrics.ic import compute_ic
        >>> from factrix.metrics.trend import ic_trend
        >>> panel = compute_forward_return(
        ...     fx.datasets.make_cs_panel(n_assets=80, n_dates=180, rng=0),
        ...     forward_periods=5,
        ... )
        >>> ic_df = compute_ic(panel)["factor"]
        >>> result = ic_trend(ic_df, value_col="ic")
        >>> result.name == ""
        True
    """
    value_col = _resolve_series_value_col(series, value_col)
    if adf_threshold is not None and not (0.0 < adf_threshold < 1.0):
        raise ValueError(
            f"adf_threshold must be a probability in (0, 1) or None, "
            f"got {adf_threshold!r}"
        )

    # Primary periods gate on the RAW date count against the stride-scaled
    # floor, matching the inspect_data pre-flight.
    sc = _enforce_scaled_floor(
        name,
        series["date"].n_unique(),
        _MIN_TREND_PERIODS,
        overlap_periods,
        "insufficient_trend_periods",
    )
    if sc is not None:
        return sc

    raw_vals = series.sort("date").drop_nulls(subset=[value_col])[value_col].to_numpy()
    n_raw_in = series.height
    n_raw = int(np.isfinite(raw_vals).sum())
    # Stride to non-overlapping observations BEFORE the test. An h-period
    # forward return makes consecutive IC observations share h-1 bars, and
    # Mann-Kendall's null variance counts every pairwise comparison as an
    # independent draw - see the Notes block for the measured cost of
    # skipping this.
    sorted_s = _sample_non_overlapping(series, overlap_periods).sort("date")
    vals = sorted_s[value_col].drop_nulls().to_numpy()
    # polars drop_nulls does not drop float NaN; an all-NaN IC series
    # (e.g. from a constant factor whose per-period rank correlation is
    # degenerate) would otherwise flow into theilslopes and _adf and
    # trip LAPACK DLASCL before any short-circuit could save us.
    vals = vals[np.isfinite(vals)]
    n = len(vals)

    # Secondary guard: null-drop can leave the strided series below the
    # effective floor even when the raw panel cleared the scaled gate.
    if n < _MIN_TREND_PERIODS:
        return _short_circuit_output(
            name,
            "insufficient_trend_periods",
            n_obs=n,
            n_obs_axis="periods",
            min_required=_MIN_TREND_PERIODS,
        )

    # WHY: index by sequence rather than date difference — non-overlapping
    # sampling can leave irregular gaps between dates.
    x = np.arange(n, dtype=float)

    result = sp_stats.theilslopes(vals, x)
    slope = float(result.slope)
    # WHY: scipy theilslopes returns (slope, intercept, low_slope, high_slope).
    low_slope = float(result.low_slope)
    high_slope = float(result.high_slope)

    # WHY: slope is significant when the CI does not cross zero.
    ci_excludes_zero = (low_slope > 0 and high_slope > 0) or (
        low_slope < 0 and high_slope < 0
    )

    # Significance from the Mann-Kendall test, the standard partner of the
    # Theil-Sen (Sen 1968) slope: both are rank-based, so the p and the
    # slope come from one framework. An earlier version backed an SE out
    # of scipy's normal-based 95% CI and read it against t(n-2) — under
    # a true null that rejected 2.4–3.8% at nominal 5% for n=10–30 where
    # Mann-Kendall sits at 4.6–5.4%. scipy's kendalltau is exact for small
    # n and asymptotic (tie-corrected) otherwise; ``tau`` is NaN only when
    # the series is constant, which ``_degenerate_test_fields`` maps to a
    # withheld test rather than a fabricated p.
    tau, p, variance_inflation = _mann_kendall_hamed_rao(vals)

    metadata: dict = {
        "stat_type": "kendall_tau",
        "h0": "tau=0",
        "method": "theil-sen slope + hamed-rao mann-kendall test",
        "n_periods": n,
        "n_periods_raw": n_raw,
        "stride": overlap_periods,
        "variance_inflation": variance_inflation,
        "ci_low": low_slope,
        "ci_high": high_slope,
        "ci_excludes_zero": ci_excludes_zero,
        "intercept": float(result.intercept),
    }
    if adf_threshold is not None:
        adf_stat, adf_p = _adf(vals)
        metadata["adf_stat"] = adf_stat
        metadata["adf_p"] = adf_p
        metadata["unit_root_suspected"] = adf_p > adf_threshold
    warning_codes: list[str] = []
    # Hamed-Rao under-corrects on a strongly persistent series (see its
    # docstring): striding removed the KNOWN overlap channel, but residual
    # persistence in the subsample still leaves the test oversized, so the
    # regime is flagged rather than silently absorbed.
    residual_autocorr = _lag1_autocorr(vals)
    metadata["residual_autocorr"] = residual_autocorr
    if residual_autocorr > PERSISTENT_SERIES_AUTOCORR:
        warning_codes.append(WarningCode.SERIAL_CORRELATION_DETECTED.value)
    # The ADF flag was metadata-only; a suspected unit root is a regime the
    # trend null is not calibrated in, so it carries a code like every other
    # regime switch in the library.
    if metadata.get("unit_root_suspected"):
        warning_codes.append(WarningCode.PERSISTENT_REGRESSOR.value)
    _surface_null_drop(
        n_periods_in=n_raw_in,
        n_periods_out=n_raw,
        drop_reason="null or non-finite value observations in the series",
        metric_name=name,
        metadata=metadata,
        warning_codes=warning_codes,
        expected_warnings=expected_warnings,
    )
    # A constant series has no rank ordering to test: tau is NaN. Keep the
    # (zero) slope and withhold the test, as every other metric does.
    stat, p_out, alternative = _degenerate_test_fields(
        tau, p, "two-sided", metadata, warning_codes
    )
    return MetricResult(
        p_value=p_out,
        alternative=alternative,
        value=slope,
        n_obs=n,
        n_obs_axis="periods",
        stat=stat,
        metadata=metadata,
        warning_codes=tuple(warning_codes),
    )
