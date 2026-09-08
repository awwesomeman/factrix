"""Series-mean inference methods — t-test on the mean of a date-indexed series.

Each member is a frozen dataclass carrying its whole ``compute`` plus
identity ClassVars (``test`` / ``se`` / ``summary``). The family shares
one date-aware input contract::

    compute(data, *, value_col, overlap_periods, alternative="two-sided") -> InferenceResult

``compute`` owns date-sort + finite-value filtering (callers pass the raw
per-period DataFrame). ``NonOverlapping`` strides the cleaned series at
``overlap_periods`` (sub-sampling away the MA(h-1) overlap), while
``NeweyWest`` / ``HansenHodrick`` keep every observation and correct the
SE via a HAC kernel. ``StationaryBootstrap`` also keeps every observation
but replaces the analytic SE with a block-bootstrap empirical p as a second
read for an adequately long, stationary series with distributional doubt; it
is not a short-sample or strong-persistence remedy. The
lag / bandwidth / block length is derived from the compute-time sample,
so the dataclasses take no *statistical* constructor knobs;
``StationaryBootstrap`` carries the two resampling knobs (``n_resamples``
/ ``rng``) that only the caller can decide.

Each member also declares, through the ``consumes_full_series``
``ClassVar``, whether it needs every period of the series or takes a
pre-strided one: a metric that strides its own panel reads that flag to
decide whether to pay for building the full overlapping series, instead
of type-checking the member.

These are metric-internal inference units: ``compute`` returns an
``InferenceResult`` whose ``stat`` / ``p_value`` feed a ``MetricResult``
directly.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from factrix._codes import WarningCode
from factrix._errors import UserInputError
from factrix._stats.bootstrap import Rng
from factrix._stats.constants import (
    MIN_PERIODS_HARD,
    MIN_PERIODS_WARN,
    PERSISTENT_SERIES_AUTOCORR,
)
from factrix._stats.diagnostics import _lag1_autocorr
from factrix._types import MIN_SERIES_PERIODS_HARD, PValueAlternative
from factrix.inference._base import InferenceResult

if TYPE_CHECKING:
    import numpy as np
    import polars as pl


_DOCS_DIRECT_COMPUTE = "api/inference#direct-compute-result"


def _validate_series_input(
    data: pl.DataFrame,
    value_col: object,
    overlap_periods: object,
    *,
    func_name: str,
) -> None:
    """Validate the shared date-indexed contract at the public boundary."""
    import polars as pl

    if isinstance(overlap_periods, bool) or not isinstance(overlap_periods, int):
        raise UserInputError(
            func_name=func_name,
            field="overlap_periods",
            value=overlap_periods,
            expected="a positive int count of periods, e.g. 5",
            docs_path=_DOCS_DIRECT_COMPUTE,
        )
    if overlap_periods <= 0:
        raise UserInputError(
            func_name=func_name,
            field="overlap_periods",
            value=overlap_periods,
            expected="a positive int count of periods (> 0)",
            docs_path=_DOCS_DIRECT_COMPUTE,
        )

    if "date" not in data.columns:
        raise UserInputError(
            func_name=func_name,
            field="date",
            value=None,
            expected="a non-null Date or Datetime column named 'date'",
            docs_path=_DOCS_DIRECT_COMPUTE,
        )
    if not isinstance(value_col, str) or value_col not in data.columns:
        raise UserInputError(
            func_name=func_name,
            field="value_col",
            value=value_col,
            candidates=data.columns,
            docs_path=_DOCS_DIRECT_COMPUTE,
        )
    value_dtype = data.schema[value_col]
    if not value_dtype.is_numeric():
        raise UserInputError(
            func_name=func_name,
            field="value_col",
            value=f"{value_col!r} has dtype {value_dtype}",
            expected=(
                "a numeric column. Encode categorical/string values or cast "
                "numeric strings before calling compute"
            ),
            docs_path=_DOCS_DIRECT_COMPUTE,
        )
    date_dtype = data.schema["date"]
    if not isinstance(date_dtype, pl.Date | pl.Datetime):
        raise UserInputError(
            func_name=func_name,
            field="date",
            value=str(date_dtype),
            expected=(
                "a non-null Date or Datetime column; parse string dates before "
                "calling compute"
            ),
            docs_path=_DOCS_DIRECT_COMPUTE,
        )
    if data["date"].null_count():
        raise UserInputError(
            func_name=func_name,
            field="date",
            value=f"{data['date'].null_count()} null row(s)",
            expected="a non-null Date or Datetime column",
            docs_path=_DOCS_DIRECT_COMPUTE,
        )

    n_dates = int(data["date"].n_unique())
    if n_dates != data.height:
        raise UserInputError(
            func_name=func_name,
            field="data",
            value=f"{data.height} rows for {n_dates} distinct dates",
            expected=(
                "exactly one observation per date. Duplicate dates have no "
                "defined order for striding or autocovariance estimation; "
                "aggregate explicitly first, e.g. "
                'data.group_by("date").mean().sort("date")'
            ),
            docs_path=_DOCS_DIRECT_COMPUTE,
        )


def _clean_series(data: pl.DataFrame, value_col: str) -> pl.Series:
    """Date-sorted finite values of ``value_col``.

    Order is fixed (sort → drop) so the stride / HAC lag math sees a
    time-coherent series regardless of caller row order. Sorting is mean-
    and OLS-invariant but load-bearing for the autocovariance terms.
    Polars ``drop_nulls`` keeps non-finite float values, and one NaN or ±Inf
    would poison the mean, HAC variance, or bootstrap centring. Keep only
    finite observations so every member reports the sample it actually tests.
    The public validator rejects non-numeric columns; the cast here normalizes
    accepted numeric dtypes before the finite-value filter, matching the metric
    series helper.
    """
    import polars as pl

    values = (
        data[value_col]
        if data["date"].is_sorted()
        else data.sort("date").get_column(value_col)
    )
    values = values.cast(pl.Float64, strict=False)
    return values.filter(values.is_finite())


def _persistent_sample(values: np.ndarray) -> bool:
    """Whether a strided subsample is persistent enough to warrant the screen.

    Two conditions, both required. The lag-1 autocorrelation must exceed
    ``PERSISTENT_SERIES_AUTOCORR``, and the subsample must reach
    ``MIN_SERIES_PERIODS_HARD`` — the library's floor for estimating a
    series statistic on the periods axis, the same one ``NonOverlapping``'s
    post-stride sample is gated on. Below it the screen is **withheld**: a
    lag-1 autocorrelation read off three to nine observations is noise, and
    a sample value above 0.3 is common there under independence, so the
    code would be reporting the shortage of periods rather than any
    persistence. Shortage of periods is what
    ``UNRELIABLE_SE_SHORT_PERIODS`` is for.
    """
    if len(values) < MIN_SERIES_PERIODS_HARD:
        return False
    return _lag1_autocorr(values) > PERSISTENT_SERIES_AUTOCORR


def _persistent_beyond_horizon(
    data: pl.DataFrame, value_col: str, overlap_periods: int
) -> bool:
    """Whether the series stays autocorrelated once the overlap is strided away.

    Overlapping ``h``-period forward returns carry an MA(h-1) structure by
    construction: lag-1 autocorrelation near ``1 - 1/h`` with lag-``h``
    near zero. That is exactly what every member of this family is built to
    absorb — the ``3(h - 1)`` bandwidth floor for the HAC kernels, the
    ``overlap_periods`` block-length floor for the bootstrap — so reading
    lag-1 off the *full* series flags the everyday overlapping case, where
    the paths are calibrated, and says nothing about the regime where they
    are not.

    The screen therefore reads lag-1 on the series strided at
    ``overlap_periods`` (first of each block, the same stride
    ``NonOverlapping`` runs its t-test on): mechanical overlap dependence is
    gone from that subsample, so what survives is persistence *beyond* the
    overlap horizon — the regime no path here is calibrated for. At
    ``overlap_periods <= 1`` the stride is a no-op and this is the plain
    lag-1 screen. Below ``MIN_SERIES_PERIODS_HARD`` strided observations the
    screen is withheld — see ``_persistent_sample``.
    """
    from factrix.metrics._helpers import _stride_dates

    strided = _clean_series(_stride_dates(data, overlap_periods), value_col).to_numpy()
    return _persistent_sample(strided)


def _persistent_array_beyond_horizon(
    values: np.ndarray, overlap_periods: int | None
) -> bool:
    """:func:`_persistent_beyond_horizon` for a date-sorted array already in hand.

    Same screen and same stride; the frame form above is for callers holding a
    ``(date, value)`` frame. Regression paths that resolve their bandwidth
    through ``_resolve_scalar_wald_hac`` run this on the *regressor* series —
    a per-period common factor persistent beyond the overlap horizon leaves
    the HAC contrast oversized in exactly the way this code reports.
    """
    import numpy as np

    values = np.asarray(values, dtype=float)
    return _persistent_sample(values[:: max(overlap_periods or 1, 1)])


@dataclass(frozen=True, slots=True)
class NonOverlapping:
    """Non-overlapping stride subsample inference: OLS t-test on every ``overlap_periods``-th observation.

    Sub-samples the cleaned series at a stride equal to ``overlap_periods``
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
    # A metric that pre-strides its own panel hands this member the strided
    # series directly; it needs no full overlapping series built for it.
    consumes_full_series: ClassVar[bool] = False

    def min_input_periods(self, overlap_periods: int) -> int:
        """Minimum input series length (periods): need ``base · h`` rows to land ``base`` after striding."""
        return MIN_SERIES_PERIODS_HARD * max(overlap_periods, 1)

    def compute(
        self,
        data: pl.DataFrame,
        *,
        value_col: str,
        overlap_periods: int,
        alternative: PValueAlternative = "two-sided",
    ) -> InferenceResult:
        from factrix._stats import _p_value_from_t, _t_stat_from_array
        from factrix._stats.core import _validate_p_value_alternative
        from factrix.metrics._helpers import _sample_non_overlapping

        _validate_series_input(
            data, value_col, overlap_periods, func_name=type(self).__name__
        )
        _validate_p_value_alternative(alternative, func_name=type(self).__name__)
        # Stride on the *calendar* (every h-th unique date) before dropping
        # non-finite rows, so a dropped observation cannot shift the sampling
        # phase; striding the cleaned row index would silently re-align the
        # subsample to overlapping windows.
        n_full = len(_clean_series(data, value_col))
        sampled = _clean_series(
            _sample_non_overlapping(data, overlap_periods), value_col
        ).to_numpy()
        n_sampled = len(sampled)

        t_stat = _t_stat_from_array(sampled)
        p_value = _p_value_from_t(t_stat, n_sampled, alternative)

        warnings: frozenset[WarningCode] = frozenset()
        # Persistence screen on the STRIDED sample — the series the t-test
        # runs on, and the same subsample the rest of the family screens
        # (see ``_persistent_beyond_horizon``); it is already in hand here,
        # so the array-level predicate is called directly. Striding an
        # AR(phi) series at h leaves autocorrelation phi^h, so a highly
        # persistent full series can hand this test a near-iid subsample:
        # AR(0.6) at h=21 sits at 4.5% (calibrated) and must not be flagged,
        # while the same series at h=1 (32.9%) must be.
        if _persistent_sample(sampled):
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
            alternative=alternative,
            estimate=float(sampled.mean()) if n_sampled else None,
            n_obs=n_sampled,
            metadata={
                "stride": overlap_periods,
                "n_obs_original": n_full,
                "n_obs_sampled": n_sampled,
            },
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class NeweyWest:
    """Newey-West HAC t-test on the full series with a Bartlett kernel.

    The bandwidth uses the [LLSW (2018)][llsw-2018] ``1.3·√T`` rule, floored
    at ``3(h - 1)`` and capped at ``T/3``. The variance is scaled by
    ``T/(T - L - 1)``, and the statistic uses effective degrees of freedom
    bounded by ``T/h - 1``. All three are derived from the input sample, so
    this class carries no lag parameter; the resolved bandwidth and degrees of
    freedom are reported as ``metadata["newey_west_lags"]`` and
    ``metadata["hac_dof"]``.

    This method retains observations that ``NonOverlapping`` drops, but it is
    not uniformly more powerful or better calibrated. Persistence that remains
    after striding at ``overlap_periods`` raises
    ``WarningCode.SERIAL_CORRELATION_DETECTED``. See the statistical-methods
    and inference-calibration references for selection guidance and measured
    limits.
    """

    test: ClassVar[str] = "t"
    se: ClassVar[str | None] = "hac"
    summary: ClassVar[str] = "Newey-West HAC t-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN
    # Every observation is kept and the dependence is corrected for, so a
    # metric that pre-strides its own panel must build the full overlapping
    # series before calling ``compute``.
    consumes_full_series: ClassVar[bool] = True

    def min_input_periods(self, overlap_periods: int) -> int:
        """Minimum input series length (periods) below which the HAC t-test cannot run."""
        return MIN_PERIODS_HARD

    def compute(
        self,
        data: pl.DataFrame,
        *,
        value_col: str,
        overlap_periods: int,
        alternative: PValueAlternative = "two-sided",
    ) -> InferenceResult:
        from factrix._stats import (
            _hac_bandwidth_ill_conditioned,
            _har_dof,
            _newey_west_t_test,
            _resolve_har_lags,
        )
        from factrix._stats.core import _validate_p_value_alternative

        _validate_series_input(
            data, value_col, overlap_periods, func_name=type(self).__name__
        )
        _validate_p_value_alternative(alternative, func_name=type(self).__name__)
        vals = _clean_series(data, value_col).to_numpy()
        n = len(vals)
        newey_west_lags = _resolve_har_lags(n, None, overlap_periods) if n >= 2 else 0
        t_stat, p_value, _ = _newey_west_t_test(
            vals,
            lags=newey_west_lags,
            overlap_periods=overlap_periods,
            alternative=alternative,
        )

        warnings: frozenset[WarningCode] = frozenset()
        # T < 5L: the kernel sum is estimated from too few lag products.
        # Structural, not just a log line (finding: method-switch-warning norm).
        if _hac_bandwidth_ill_conditioned(n, newey_west_lags):
            warnings |= frozenset({WarningCode.HAC_BANDWIDTH_ILL_CONDITIONED})
        # Persistence screen: read on the series strided at overlap_periods,
        # so the MA(h-1) overlap this member is built to absorb does not trip
        # it and only persistence beyond the overlap horizon — where no
        # member of this family is calibrated — does (see
        # ``_persistent_beyond_horizon``, WarningCode.SERIAL_CORRELATION_DETECTED).
        if _persistent_beyond_horizon(data, value_col, overlap_periods):
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
            alternative=alternative,
            metadata={
                "newey_west_lags": newey_west_lags,
                "hac_dof": _har_dof(n, newey_west_lags, overlap_periods)
                if n >= 3
                else None,
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
    # Every observation is kept and the dependence is corrected for, so a
    # metric that pre-strides its own panel must build the full overlapping
    # series before calling ``compute``.
    consumes_full_series: ClassVar[bool] = True

    def min_input_periods(self, overlap_periods: int) -> int:
        """Minimum input series length (periods) below which the HAC t-test cannot run."""
        return MIN_PERIODS_HARD

    def compute(
        self,
        data: pl.DataFrame,
        *,
        value_col: str,
        overlap_periods: int,
        alternative: PValueAlternative = "two-sided",
    ) -> InferenceResult:
        from factrix._stats import _hansen_hodrick_t_test
        from factrix._stats.core import _validate_p_value_alternative

        _validate_series_input(
            data, value_col, overlap_periods, func_name=type(self).__name__
        )
        _validate_p_value_alternative(alternative, func_name=type(self).__name__)
        vals = _clean_series(data, value_col).to_numpy()
        n = len(vals)
        t_stat, p_value, _, clamped = _hansen_hodrick_t_test(
            vals, overlap_periods=overlap_periods, alternative=alternative
        )

        warnings: frozenset[WarningCode] = frozenset()
        # Persistence screen: read on the series strided at overlap_periods,
        # so the MA(h-1) overlap this member is built to absorb does not trip
        # it and only persistence beyond the overlap horizon — where no
        # member of this family is calibrated — does (see
        # ``_persistent_beyond_horizon``, WarningCode.SERIAL_CORRELATION_DETECTED).
        if _persistent_beyond_horizon(data, value_col, overlap_periods):
            warnings |= frozenset({WarningCode.SERIAL_CORRELATION_DETECTED})
        if clamped:
            warnings |= frozenset({WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE})
        # As in ``NeweyWest``: only a NaN from a sample the kernel could
        # actually run on is degeneracy rather than a shortage.
        if math.isnan(t_stat) and n >= 3 and overlap_periods >= 1:
            warnings |= frozenset({WarningCode.DEGENERATE_VARIANCE})
        if 0 < n < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        # ``estimate`` / ``n_obs`` were omitted here alone; all three siblings
        # populate them, so a caller reading the harmonized point estimate or
        # sample size off this member got None.
        return InferenceResult(
            stat=t_stat,
            p_value=p_value,
            alternative=alternative,
            metadata={"kernel": "rectangular", "variance_clamped": clamped},
            warnings=warnings,
            estimate=float(vals.mean()) if n else None,
            n_obs=n,
        )


@dataclass(frozen=True, slots=True)
class StationaryBootstrap:
    r"""Stationary-bootstrap empirical-p inference on a series mean.

    Resamples geometric-length blocks ([Politis-Romano 1994][politis-romano-1994])
    from the series, centred under $H_0: \mathbb{E}[x] = 0$, and reports the
    empirical p-value from a studentized bootstrap-t root. Block length is
    selected automatically per [Politis-White (2004)][politis-white-2004] and
    cannot be shorter than ``overlap_periods``. This is a second read for an
    adequately long, stationary series with distributional doubt; it is not a
    short-sample or strong-persistence remedy.

    Metadata records the resolved block length, seed, usable resample count,
    and Monte Carlo standard error. If studentization is impossible, the method
    uses a raw-mean root and raises ``WarningCode.DEGENERATE_VARIANCE``. See the
    inference-calibration reference for measured limits.

    Args:
        n_resamples: Number of bootstrap draws. Must be at least
            ``BOOTSTRAP_RESAMPLES_FLOOR``; defaults to 999.
        rng: An ``int``, ``None``, or a ``numpy.random.Generator``.
            ``None`` resolves and reports a seed. A ``Generator`` is advanced
            in place and reports no seed because the caller owns the stream.
    """

    n_resamples: int = 999
    rng: Rng = None

    test: ClassVar[str] = "bootstrap-mean"
    se: ClassVar[str | None] = "bootstrap"
    summary: ClassVar[str] = "stationary-bootstrap empirical p-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN
    # Every observation is kept and the dependence is corrected for, so a
    # metric that pre-strides its own panel must build the full overlapping
    # series before calling ``compute``.
    consumes_full_series: ClassVar[bool] = True

    def __post_init__(self) -> None:
        from factrix._stats.bootstrap import _check_n_resamples

        _check_n_resamples(
            self.n_resamples,
            func_name="StationaryBootstrap",
            docs_path="reference/statistical-methods",
        )

    def min_input_periods(self, overlap_periods: int) -> int:
        """Minimum input series length (periods); no overlap-specific floor."""
        return MIN_PERIODS_HARD

    def compute(
        self,
        data: pl.DataFrame,
        *,
        value_col: str,
        overlap_periods: int,
        alternative: PValueAlternative = "two-sided",
    ) -> InferenceResult:
        from factrix._stats.bootstrap import _block_bootstrap_diff_p
        from factrix._stats.core import _validate_p_value_alternative

        _validate_series_input(
            data, value_col, overlap_periods, func_name=type(self).__name__
        )
        _validate_p_value_alternative(alternative, func_name=type(self).__name__)
        vals = _clean_series(data, value_col).to_numpy()
        n = len(vals)
        p_value, boot_metadata = _block_bootstrap_diff_p(
            vals,
            n_resamples=self.n_resamples,
            overlap_periods=overlap_periods,
            alternative=alternative,
            rng=self.rng,
        )

        warnings: frozenset[WarningCode] = frozenset()
        # Persistence screen: read on the series strided at overlap_periods,
        # so the MA(h-1) overlap this member is built to absorb does not trip
        # it and only persistence beyond the overlap horizon — where no
        # member of this family is calibrated — does (see
        # ``_persistent_beyond_horizon``, WarningCode.SERIAL_CORRELATION_DETECTED).
        if _persistent_beyond_horizon(data, value_col, overlap_periods):
            warnings |= frozenset({WarningCode.SERIAL_CORRELATION_DETECTED})
        if 0 < n < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})
        # The kernel drops from the bootstrap-t root to the raw-mean root
        # when it cannot form a block SE — a sample-driven method switch, so
        # it must not be silent. Above n=2 the only way to get there is a
        # sample with no usable dispersion, which is what the code names.
        if n >= 2 and boot_metadata.get("studentized") is False:
            warnings |= frozenset({WarningCode.DEGENERATE_VARIANCE})

        return InferenceResult(
            stat=float(vals.mean()) if n else float("nan"),
            p_value=p_value,
            alternative=alternative,
            metadata=dict(boot_metadata),
            warnings=warnings,
            estimate=float(vals.mean()) if n else None,
            n_obs=n,
        )


NON_OVERLAPPING = NonOverlapping()
NEWEY_WEST = NeweyWest()
HANSEN_HODRICK = HansenHodrick()
STATIONARY_BOOTSTRAP = StationaryBootstrap()
