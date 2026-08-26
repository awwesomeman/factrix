"""Series-mean inference methods — t-test on the mean of a date-indexed series.

Each member is a frozen dataclass carrying its whole ``compute`` plus
identity ClassVars (``test`` / ``se`` / ``summary``). The family shares
one date-aware input contract::

    compute(data: pl.DataFrame, *, value_col: str, forward_periods: int) -> InferenceResult

``compute`` owns date-sort + null-drop (callers pass the raw per-period
DataFrame). ``NonOverlapping`` strides the cleaned series at
``forward_periods`` (sub-sampling away the MA(h-1) overlap), while
``NeweyWest`` / ``HansenHodrick`` keep every observation and correct the
SE via a HAC kernel. ``StationaryBootstrap`` also keeps every observation
but replaces the analytic SE with a block-bootstrap empirical p, for
series too short or non-normal for a HAC t-test to be trusted. The
lag / bandwidth / block length is derived from the compute-time sample,
so the dataclasses take no constructor knobs.

These are metric-internal inference units: ``compute`` returns an
``InferenceResult`` whose ``stat`` / ``p_value`` feed a ``MetricResult``
directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from factrix._codes import WarningCode
from factrix._stats.constants import (
    MIN_PERIODS_HARD,
    MIN_PERIODS_WARN,
    PERSISTENT_SERIES_AUTOCORR,
)
from factrix._stats.diagnostics import _lag1_autocorr
from factrix._types import MIN_SERIES_PERIODS_HARD
from factrix.inference._base import InferenceResult

if TYPE_CHECKING:
    import polars as pl


def _clean_series(data: pl.DataFrame, value_col: str) -> pl.Series:
    """Date-sorted values of ``value_col`` with nulls *and* NaNs dropped.

    Order is fixed (sort → drop) so the stride / HAC lag math sees a
    time-coherent series regardless of caller row order. Sorting is mean-
    and OLS-invariant but load-bearing for the autocovariance terms.
    polars ``drop_nulls`` keeps float NaN, and a single NaN would poison the
    mean, the HAC variance and the bootstrap centring, so it is dropped too.
    """
    if data["date"].is_sorted():
        return data[value_col].drop_nulls().drop_nans()
    return data.sort("date").get_column(value_col).drop_nulls().drop_nans()


@dataclass(frozen=True, slots=True)
class NonOverlapping:
    """Non-overlapping stride subsample inference: OLS t-test on every ``forward_periods``-th observation.

    Sub-samples the cleaned series at a stride equal to ``forward_periods``
    to break the MA(h-1) autocorrelation that overlapping h-period forward
    returns induce ([Hansen-Hodrick 1980][hansen-hodrick-1980]), then runs
    a textbook OLS t-test on the survivors. The most conservative
    overlap-aware path — it discards h-1 of every h observations rather
    than correcting the SE; ``NeweyWest`` is the less-lossy HAC
    alternative on the full series.
    """

    test: ClassVar[str] = "t"
    se: ClassVar[str | None] = "ols"
    summary: ClassVar[str] = "non-overlapping t-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN

    def min_input_periods(self, forward_periods: int) -> int:
        """Minimum input series length (periods): need ``base · h`` rows to land ``base`` after striding."""
        return MIN_SERIES_PERIODS_HARD * max(forward_periods, 1)

    def compute(
        self, data: pl.DataFrame, *, value_col: str, forward_periods: int
    ) -> InferenceResult:
        from factrix._stats import _p_value_from_t, _t_stat_from_array
        from factrix.metrics._helpers import _sample_non_overlapping

        # Stride on the *calendar* (every h-th unique date) before dropping
        # non-finite rows, so a dropped observation cannot shift the sampling
        # phase; striding the cleaned row index would silently re-align the
        # subsample to overlapping windows.
        n_full = len(_clean_series(data, value_col))
        sampled = _clean_series(
            _sample_non_overlapping(data, forward_periods), value_col
        ).to_numpy()
        n_sampled = len(sampled)

        t_stat = _t_stat_from_array(sampled)
        p_value = _p_value_from_t(t_stat, n_sampled)

        warnings: frozenset[WarningCode] = frozenset()
        # Persistence screen on the STRIDED sample — the series the t-test
        # runs on. Striding an AR(phi) series at h leaves autocorrelation
        # phi^h, so a highly persistent full series can hand this test a
        # near-iid subsample: AR(0.6) at h=21 sits at 4.5% (calibrated) and
        # must not be flagged, while the same series at h=1 (32.9%) must be.
        if _lag1_autocorr(sampled) > PERSISTENT_SERIES_AUTOCORR:
            warnings |= frozenset({WarningCode.SERIAL_CORRELATION_DETECTED})
        # A NaN t on a subsample long enough to test means no dispersion at
        # all (every survivor identical). Flag it rather than let a NaN p read
        # as a merely uninformative result. Below two survivors the NaN is a
        # data shortage, not degeneracy — UNRELIABLE_SE_SHORT_PERIODS covers
        # that — so it must not carry this code.
        if math.isnan(t_stat) and n_sampled >= 2:
            warnings |= frozenset({WarningCode.DEGENERATE_VARIANCE})
        if 0 < n_sampled < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_stat,
            p_value=p_value,
            estimate=float(sampled.mean()) if n_sampled else None,
            n_obs=n_sampled,
            metadata={
                "stride": forward_periods,
                "n_obs_original": n_full,
                "n_obs_sampled": n_sampled,
            },
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class NeweyWest:
    """Newey-West (1987) HAC SE inference: t-test on the full series with a Bartlett-kernel HAC variance.

    Keeps every observation and absorbs the autocorrelation induced by
    overlapping ``forward_periods``-period returns through HAC standard
    errors rather than dropping samples. Bandwidth is the
    [LLSW (2018)][llsw-2018] HAR rule ``1.3·√T`` floored at ``3(h - 1)``
    and capped at ``T/3``; the SE carries the ``T/(T - L - 1)``
    finite-sample scale and the t is read against effective df
    ``min(1.5·T/L - 1, T/h - 1)`` rather than ``T - 1``. All three are
    derived from the compute-time sample, so the dataclass carries no
    lag knob. See ``factrix._stats.hac._newey_west_t_test`` for the
    measured size table.

    Not a strict upgrade over ``NonOverlapping``. ``NonOverlapping`` is
    calibrated in every overlapping cell measured (4.5–5.4%) at the cost
    of ``h-1`` of every ``h`` observations; ``NeweyWest`` keeps the whole
    sample and measures 3.9–7.3% across ``T ∈ {60, 120, 240, 500} ×
    h ∈ {1, 5, 21}``. Prefer ``NeweyWest`` for power on short samples and
    ``NonOverlapping`` when size discipline matters more than power.
    Neither is calibrated above ``PERSISTENT_SERIES_AUTOCORR``.
    """

    test: ClassVar[str] = "t"
    se: ClassVar[str | None] = "hac"
    summary: ClassVar[str] = "Newey-West HAC t-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN

    def min_input_periods(self, forward_periods: int) -> int:
        """Minimum input series length (periods) below which the HAC t-test cannot run."""
        return MIN_PERIODS_HARD

    def compute(
        self, data: pl.DataFrame, *, value_col: str, forward_periods: int
    ) -> InferenceResult:
        from factrix._stats import (
            _hac_bandwidth_ill_conditioned,
            _har_dof,
            _newey_west_t_test,
            _resolve_har_lags,
        )

        vals = _clean_series(data, value_col).to_numpy()
        n = len(vals)
        nw_lags = _resolve_har_lags(n, None, forward_periods) if n >= 2 else 0
        t_stat, p_value, _ = _newey_west_t_test(
            vals, lags=nw_lags, forward_periods=forward_periods
        )

        warnings: frozenset[WarningCode] = frozenset()
        # T < 5L: the kernel sum is estimated from too few lag products.
        # Structural, not just a log line (finding: method-switch-warning norm).
        if _hac_bandwidth_ill_conditioned(n, nw_lags):
            warnings |= frozenset({WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED})
        # Persistence screen: above PERSISTENT_SERIES_AUTOCORR no member of
        # this family is calibrated (see WarningCode.SERIAL_CORRELATION_DETECTED).
        if _lag1_autocorr(vals) > PERSISTENT_SERIES_AUTOCORR:
            warnings |= frozenset({WarningCode.SERIAL_CORRELATION_DETECTED})
        # ``n < 3`` is a shortage the kernel cannot run on, flagged by
        # UNRELIABLE_SE_SHORT_PERIODS; only a NaN above that floor is a
        # collapsed HAC SE.
        if math.isnan(t_stat) and n >= 3:
            warnings |= frozenset({WarningCode.DEGENERATE_VARIANCE})
        if 0 < n < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_stat,
            p_value=p_value,
            metadata={
                "nw_lags": nw_lags,
                "hac_dof": _har_dof(n, nw_lags, forward_periods) if n >= 3 else None,
            },
            warnings=warnings,
            estimate=float(vals.mean()) if n else None,
            n_obs=n,
        )


@dataclass(frozen=True, slots=True)
class HansenHodrick:
    """Hansen-Hodrick (1980) rectangular-kernel HAC SE inference on a series mean.

    Closed-form rectangular-kernel HAC variance matched to the MA(h-1)
    overlap structure of h-period forward returns. No PSD guarantee
    ([Andrews 1991][andrews-1991] §3): on short / mildly anti-correlated
    samples the estimate can come out negative; ``compute`` clamps the
    variance to 0 and surfaces ``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE``.
    A clamped (or otherwise zero) SE leaves no computable t, so ``stat`` /
    ``p_value`` are NaN and ``WarningCode.DEGENERATE_VARIANCE`` is raised
    alongside.

    Exported for explicit / comparison use but **not** in any metric's
    ``inference=`` union today: ``NeweyWest`` (Bartlett, PSD-guaranteed) is
    the recommended HAC, and the spread-metric dispatch is ``NeweyWest``-
    specific. See ``factrix.inference``'s module docstring for the full
    rationale.
    """

    test: ClassVar[str] = "t"
    se: ClassVar[str | None] = "hac"
    summary: ClassVar[str] = "Hansen-Hodrick HAC t-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN

    def min_input_periods(self, forward_periods: int) -> int:
        """Minimum input series length (periods) below which the HAC t-test cannot run."""
        return MIN_PERIODS_HARD

    def compute(
        self, data: pl.DataFrame, *, value_col: str, forward_periods: int
    ) -> InferenceResult:
        from factrix._stats import _hansen_hodrick_t_test

        vals = _clean_series(data, value_col).to_numpy()
        t_stat, p_value, _, clamped = _hansen_hodrick_t_test(
            vals, forward_periods=forward_periods
        )

        warnings: frozenset[WarningCode] = frozenset()
        # Persistence screen: above PERSISTENT_SERIES_AUTOCORR no member of
        # this family is calibrated (see WarningCode.SERIAL_CORRELATION_DETECTED).
        if _lag1_autocorr(vals) > PERSISTENT_SERIES_AUTOCORR:
            warnings |= frozenset({WarningCode.SERIAL_CORRELATION_DETECTED})
        if clamped:
            warnings |= frozenset({WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE})
        # As in ``NeweyWest``: only a NaN from a sample the kernel could
        # actually run on is degeneracy rather than a shortage.
        if math.isnan(t_stat) and len(vals) >= 3 and forward_periods >= 1:
            warnings |= frozenset({WarningCode.DEGENERATE_VARIANCE})
        if 0 < len(vals) < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_stat,
            p_value=p_value,
            metadata={"kernel": "rectangular", "variance_clamped": clamped},
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class StationaryBootstrap:
    r"""Stationary-bootstrap empirical-p inference on a series mean.

    Resamples geometric-length blocks ([Politis-Romano 1994][politis-romano-1994])
    from the series, centred under $H_0: \mathbb{E}[x] = 0$, and reports the
    two-sided empirical p — the fraction of bootstrap means at least as
    extreme as the observed one (Davison-Hinkley ``+1`` smoothing). No
    normality or asymptotic-variance assumption, unlike ``NeweyWest`` /
    ``HansenHodrick``: appropriate when the series is short relative to its
    dependence horizon or heavy-tailed / skewed enough that a HAC t-test is
    unreliable. Block length resolves automatically per
    [Politis-White (2004)][politis-white-2004]; the resolved seed is
    reported in ``metadata`` so the run is reproducible after the fact even
    though the dataclass itself carries no seed knob.

    Delegates to ``factrix._stats.bootstrap._block_bootstrap_diff_p`` —
    the same kernel backing ``factrix.stats.BlockBootstrap`` — so the
    empirical-p convention is one implementation, not a parallel one.
    """

    test: ClassVar[str] = "bootstrap-mean"
    se: ClassVar[str | None] = "bootstrap"
    summary: ClassVar[str] = "stationary-bootstrap empirical p-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN

    def min_input_periods(self, forward_periods: int) -> int:
        """Minimum input series length (periods); no overlap-specific floor."""
        return MIN_PERIODS_HARD

    def compute(
        self, data: pl.DataFrame, *, value_col: str, forward_periods: int
    ) -> InferenceResult:
        from factrix._stats.bootstrap import _block_bootstrap_diff_p

        vals = _clean_series(data, value_col).to_numpy()
        n = len(vals)
        p_value, boot_metadata = _block_bootstrap_diff_p(vals)

        warnings: frozenset[WarningCode] = frozenset()
        # Persistence screen: above PERSISTENT_SERIES_AUTOCORR no member of
        # this family is calibrated (see WarningCode.SERIAL_CORRELATION_DETECTED).
        if _lag1_autocorr(vals) > PERSISTENT_SERIES_AUTOCORR:
            warnings |= frozenset({WarningCode.SERIAL_CORRELATION_DETECTED})
        if 0 < n < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=float(vals.mean()) if n else float("nan"),
            p_value=p_value,
            metadata=dict(boot_metadata),
            warnings=warnings,
            estimate=float(vals.mean()) if n else None,
            n_obs=n,
        )


NON_OVERLAPPING = NonOverlapping()
NEWEY_WEST = NeweyWest()
HANSEN_HODRICK = HansenHodrick()
STATIONARY_BOOTSTRAP = StationaryBootstrap()
