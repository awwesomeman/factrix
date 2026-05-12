"""``GMM`` MomentEstimator — Hansen (1982) two-step efficient J-test.

Names the over-identifying-restriction inference path emitted to
``FactorProfile.stats`` as ``StatCode.P_GMM`` (J-statistic key
``StatCode.J_GMM`` lands together with the multi-horizon panel cell
procedure that emits it). ``compute(moments, *, forward_periods)``
delegates to :func:`factrix._stats._two_step_gmm_j_stat` so cell
procedures share one path with the standalone primitive.

Pure over-identification (``n_params = 0``) is the only mode supported
in this release — parametric GMM (common-mean / shared-β restrictions)
is a forward hook on the underlying primitive and on this class.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from factrix._axis import FactorScope, Metric, Signal
from factrix._codes import StatCode, WarningCode
from factrix._stats.constants import MIN_PERIODS_WARN
from factrix.stats._estimator import GMMResult

if TYPE_CHECKING:
    import numpy as np


@dataclass(frozen=True, slots=True)
class GMM:
    """Hansen (1982) two-step efficient GMM J-test estimator.

    Computes the J-statistic for a moment-condition system whose
    null is ``E[g] = 0`` (pure over-identification). The long-run
    covariance uses the Bartlett kernel sharing the Newey-West
    bandwidth convention with ``NeweyWest`` so the
    ``forward_periods - 1`` overlap floor is uniform across HAC and
    GMM inference.

    Applicability is restricted to ``(INDIVIDUAL, CONTINUOUS)`` cells
    — the multi-horizon forward-return panel is the first cell that
    dispatches this estimator; multi-bucket and cross-sectional
    shared-β moment systems are deferred to follow-up work.

    Pass an instance to ``AnalysisConfig`` (via ``moment_estimator=``)
    to drive evaluate-time inference once cell dispatch lands::

        cfg = AnalysisConfig.individual_continuous(
            metric=Metric.IC, moment_estimator=GMM(),
        )

    Solver tuning lives on the dataclass: ``max_iter`` caps iterations
    beyond step 2 (no effect for pure overid since there is no
    parameter to update — exposed as a forward hook for the parametric
    extension).
    """

    max_iter: int = 2

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return (
            "Hansen (1982) two-step efficient GMM J-test for over-"
            "identifying moment restrictions (Bartlett-kernel long-run "
            "covariance, NW1987 bandwidth with overlap floor)."
        )

    @property
    def min_periods(self) -> int:
        return MIN_PERIODS_WARN

    def applicable_to(self, scope: FactorScope, signal: Signal) -> bool:
        return scope is FactorScope.INDIVIDUAL and signal is Signal.CONTINUOUS

    def emits_for(
        self,
        _scope: FactorScope,
        _signal: Signal,
        _metric: Metric | None,
    ) -> StatCode:
        return StatCode.P_GMM

    def compute(
        self,
        moments: np.ndarray,
        *,
        forward_periods: int,
    ) -> GMMResult:
        from scipy import stats as sp_stats

        from factrix._stats import _two_step_gmm_j_stat

        j_stat, df, n_iter, weight_singular = _two_step_gmm_j_stat(
            moments,
            forward_periods=forward_periods,
            max_iter=self.max_iter,
        )
        overid_p = float(sp_stats.chi2.sf(j_stat, df=df))

        warnings: frozenset[WarningCode] = frozenset()
        if weight_singular:
            warnings |= frozenset({WarningCode.SINGULAR_WEIGHT_MATRIX})
        if 0 < moments.shape[0] < self.min_periods:
            warnings |= frozenset({WarningCode.UNRELIABLE_SE_SHORT_PERIODS})

        return GMMResult(
            j_stat=j_stat,
            df=df,
            overid_p=overid_p,
            n_moments=moments.shape[1],
            n_params=0,
            metadata={
                "weight_matrix_iter": n_iter,
                "weight_singular": weight_singular,
            },
            warnings=warnings,
        )
