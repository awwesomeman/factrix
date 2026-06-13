"""Series-mean inference methods — t-test on the mean of a date-indexed series.

Each member is a frozen dataclass carrying its whole ``compute`` plus
identity ClassVars (``test`` / ``se`` / ``summary``). The family shares
one date-aware input contract::

    compute(df: pl.DataFrame, *, value_col: str, forward_periods: int) -> InferenceResult

``compute`` owns date-sort + null-drop (callers pass the raw per-date
DataFrame). ``NonOverlapping`` strides the cleaned series at
``forward_periods`` (sub-sampling away the MA(h-1) overlap), while
``NeweyWest`` / ``HansenHodrick`` keep every observation and correct the
SE via a HAC kernel. The lag / bandwidth is derived from the
compute-time sample, so the dataclasses take no constructor knobs.

These are metric-internal inference units: ``compute`` returns
``InferenceResult`` with ``stat_name`` / ``p_name`` ``None`` (no
``profile.stats`` StatCode claimed). Promotion to a StatCode-emitting
profile estimator is the structured-shape redesign tracked separately.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, ClassVar

from factrix._codes import WarningCode
from factrix._stats.constants import MIN_PERIODS_HARD, MIN_PERIODS_WARN
from factrix._types import MIN_ASSETS_PER_DATE_IC
from factrix.inference._base import InferenceResult

if TYPE_CHECKING:
    import polars as pl


def _clean_series(df: pl.DataFrame, value_col: str) -> pl.Series:
    """Date-sorted, null-dropped values of ``value_col``.

    Order is fixed (sort → drop-null) so the stride / HAC lag math sees a
    time-coherent series regardless of caller row order. Sorting is mean-
    and OLS-invariant but load-bearing for the autocovariance terms.
    """
    return df.sort("date").drop_nulls(value_col)[value_col]


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

    def hard_floor(self, forward_periods: int) -> int:
        """Raw-period floor: need ``base · h`` rows to land ``base`` after striding."""
        return MIN_ASSETS_PER_DATE_IC * max(forward_periods, 1)

    def compute(
        self, df: pl.DataFrame, *, value_col: str, forward_periods: int
    ) -> InferenceResult:
        from factrix._stats import _p_value_from_t, _t_stat_from_array

        vals = _clean_series(df, value_col).to_numpy()
        sampled = vals[::forward_periods]
        n_sampled = len(sampled)

        t_stat = _t_stat_from_array(sampled)
        p_value = _p_value_from_t(t_stat, n_sampled)

        warnings: frozenset[WarningCode] = frozenset()
        if 0 < n_sampled < self.min_periods:
            warnings = frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_stat,
            p_value=p_value,
            stat_name=None,
            p_name=None,
            metadata={
                "stride": forward_periods,
                "n_obs_original": len(vals),
                "n_obs_sampled": n_sampled,
            },
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class NeweyWest:
    """Newey-West (1987) HAC SE inference: t-test on the full series with a Bartlett-kernel HAC variance.

    Keeps every observation and absorbs the autocorrelation induced by
    overlapping ``forward_periods``-day returns through HAC standard
    errors rather than dropping samples. Bandwidth is the
    [Newey-West (1994)][newey-west-1994] automatic Bartlett rule
    (``auto_bartlett``) floored at ``forward_periods - 1`` for the
    induced MA(h-1) overlap; it is derived from the compute-time sample,
    so the dataclass carries no lag knob.
    """

    test: ClassVar[str] = "t"
    se: ClassVar[str | None] = "hac"
    summary: ClassVar[str] = "Newey-West HAC t-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN

    def hard_floor(self, forward_periods: int) -> int:
        """Raw-period floor below which the HAC t-test cannot run."""
        return MIN_PERIODS_HARD

    def compute(
        self, df: pl.DataFrame, *, value_col: str, forward_periods: int
    ) -> InferenceResult:
        from factrix._stats import _newey_west_t_test, _resolve_nw_lags
        from factrix._stats.constants import auto_bartlett

        vals = _clean_series(df, value_col).to_numpy()
        n = len(vals)
        nw_lags = (
            _resolve_nw_lags(n, auto_bartlett(n), forward_periods) if n >= 2 else 0
        )
        t_stat, p_value, _ = _newey_west_t_test(vals, lags=nw_lags)

        warnings: frozenset[WarningCode] = frozenset()
        if 0 < n < self.min_periods:
            warnings = frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_stat,
            p_value=p_value,
            stat_name=None,
            p_name=None,
            metadata={"nw_lags": nw_lags},
            warnings=warnings,
        )


@dataclass(frozen=True, slots=True)
class HansenHodrick:
    """Hansen-Hodrick (1980) rectangular-kernel HAC SE inference on a series mean.

    Closed-form rectangular-kernel HAC variance matched to the MA(h-1)
    overlap structure of h-period forward returns. No PSD guarantee
    ([Andrews 1991][andrews-1991] §3): on short / mildly anti-correlated
    samples the estimate can come out negative; ``compute`` clamps the
    variance to 0 and surfaces ``WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE``.
    """

    test: ClassVar[str] = "t"
    se: ClassVar[str | None] = "hac"
    summary: ClassVar[str] = "Hansen-Hodrick HAC t-test"
    min_periods: ClassVar[int] = MIN_PERIODS_WARN

    def hard_floor(self, forward_periods: int) -> int:
        """Raw-period floor below which the HAC t-test cannot run."""
        return MIN_PERIODS_HARD

    def compute(
        self, df: pl.DataFrame, *, value_col: str, forward_periods: int
    ) -> InferenceResult:
        from factrix._stats import _hansen_hodrick_t_test

        vals = _clean_series(df, value_col).to_numpy()
        t_stat, p_value, _, clamped = _hansen_hodrick_t_test(
            vals, forward_periods=forward_periods
        )

        warnings: frozenset[WarningCode] = frozenset()
        if clamped:
            warnings |= frozenset({WarningCode.RECT_KERNEL_NEGATIVE_VARIANCE})
        if 0 < len(vals) < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return InferenceResult(
            stat=t_stat,
            p_value=p_value,
            stat_name=None,
            p_name=None,
            metadata={"kernel": "rectangular", "variance_clamped": clamped},
            warnings=warnings,
        )


NON_OVERLAPPING = NonOverlapping()
NEWEY_WEST = NeweyWest()
HANSEN_HODRICK = HansenHodrick()
