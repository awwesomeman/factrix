"""``NonOverlappingSample`` — OLS t-test on a non-overlapping stride subsample of a series.

A metric-internal inference unit, not a registered ``Estimator``: it
emits no ``StatCode`` and does not enter ``_ESTIMATOR_REGISTRY`` /
``list_estimators`` / the ``profile.stats`` dispatch path. ``ic()``
delegates its default-path significance test here so the math lives in
one named, independently-testable place instead of being inlined in the
metric body.

Promotion to a first-class ``Estimator`` (with a ``(stat, p)`` StatCode
pair and ``emits_for`` dispatch) is gated behind the structured-shape
redesign the ``StatCode`` enum note requires before the codebook grows
further — see ``factrix._codes.StatCode``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from factrix._codes import WarningCode
from factrix._stats.constants import MIN_PERIODS_WARN
from factrix.stats._estimator import InferenceResult

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class NonOverlappingSample:
    """Non-overlapping stride subsample inference: OLS t-test on every ``forward_periods``-th observation.

    Sub-samples the series at a stride equal to ``forward_periods`` to
    break the MA(h-1) autocorrelation that overlapping h-period forward
    returns induce ([Hansen-Hodrick 1980][hansen-hodrick-1980]), then
    runs a textbook OLS t-test on the surviving observations. The most
    conservative overlap-aware path — it discards h-1 of every h
    observations rather than correcting the SE; ``NeweyWest`` is the
    less-lossy HAC alternative on the full series.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return (
            "OLS t-test on non-overlapping stride subsamples (stride = forward_periods) "
            "→ t → two-sided p-value."
        )

    @property
    def min_periods(self) -> int:
        """Soft floor on the *sampled* count below which the SE is frail.

        Surfaced via ``warnings`` rather than silently degrading; the
        hard short-circuit (sample too short to test at all) is the
        caller's responsibility.
        """
        return MIN_PERIODS_WARN

    def compute(
        self,
        series: np.ndarray,
        *,
        forward_periods: int,
    ) -> InferenceResult:
        """Run the non-overlapping OLS t-test on ``series``.

        Args:
            series: 1-D test-target series (e.g. the per-date IC series),
                already null-dropped and ordered by the caller. The unit
                strides it at ``forward_periods``; it does not re-sort or
                re-clean.
            forward_periods: Sub-sampling stride (the overlap horizon of
                the underlying forward returns).

        Returns:
            ``InferenceResult`` with ``stat`` / ``p_value`` and
            ``stat_name`` / ``p_name`` set to ``None`` (no profile
            StatCode claimed). ``metadata`` carries ``stride`` and the
            pre / post-sampling counts.
        """
        from factrix._stats import _p_value_from_t, _t_stat_from_array

        sampled = series[::forward_periods]
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
                "n_obs_original": len(series),
                "n_obs_sampled": n_sampled,
            },
            warnings=warnings,
        )
