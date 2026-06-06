"""``DriscollKraay`` Estimator — cross-section-robust HAC SE for pooled-panel slopes (#537).

Names the [Driscoll & Kraay (1998)][driscoll-kraay-1998] cross-sectional
dependence-robust standard-error path. Selection-only base
``Estimator`` (like ``WaldNWCluster`` / ``WaldTwoWayCluster``): the
numerics live in ``factrix._stats.hac._driscoll_kraay_cov`` and are
driven from inside ``factrix.metrics.fm_beta.pooled_beta`` via the
lowercase ``factrix.estimators.driscoll_kraay`` callable, not through a
``HACEstimator.compute`` on a 1-D series — DK consumes the full
``(date, asset)`` panel, not a mean.

Applicability is restricted to ``(INDIVIDUAL, DENSE)`` — the pooled-OLS
cell ``pooled_beta`` runs on. ``COMMON`` cells collapse to one value per
date and have no within-period cross-section for DK to aggregate over.

``emits_for`` returns the singleton ``StatCode.P_DK``: ``pooled_beta``
reports its t-stat inside its own ``MetricResult.metadata`` rather than
in ``profile.stats``, so — like ``WaldTwoWayCluster`` — this is a
reserved interface surfaced by ``list_estimators`` for discovery; a
``bhy(estimator=DriscollKraay())`` call against an ``evaluate()`` profile
lands on the missing-stat path until a function populates
``profile.stats[StatCode.P_DK]``.
"""

from __future__ import annotations

from factrix._axis import FactorDensity, FactorScope
from factrix._codes import StatCode


class DriscollKraay:
    """[Driscoll & Kraay (1998)][driscoll-kraay-1998] cross-section-robust heteroskedasticity-and-autocorrelation-consistent (HAC) SE for a pooled-panel slope.

    Aggregates the per-observation OLS scores cross-sectionally within
    each period, then runs a Bartlett-kernel HAC on the resulting time
    series of cross-sectional sums and sandwiches with ``(X'X)⁻¹``. The
    result is robust to **arbitrary contemporaneous cross-sectional
    correlation** — the gap a one-way cluster-on-date SE leaves open,
    which understates SE for small, cross-sectionally correlated panels
    ([Petersen (2009)][petersen-2009]). Numerics live in
    ``factrix._stats.hac._driscoll_kraay_cov``; ``pooled_beta`` drives
    them via the lowercase ``factrix.estimators.driscoll_kraay`` callable.

    Pass an instance to a function to make the inference choice explicit;
    surfaced by ``list_estimators`` on the ``(INDIVIDUAL, DENSE)`` cell.
    Constructor takes no arguments in this release; the Bartlett bandwidth
    is resolved automatically ([Newey-West (1994)][newey-west-1994]
    ``auto_bartlett`` on the period count) and an explicit-lag override
    lives on the ``pooled_beta`` call rather than on the instance.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return (
            "Driscoll-Kraay (1998) cross-section-robust HAC SE (Bartlett "
            "kernel on cross-sectional score sums) → t → two-sided p-value "
            "for a pooled-panel slope."
        )

    def applicable_to(self, scope: FactorScope, density: FactorDensity) -> bool:
        return scope is FactorScope.INDIVIDUAL and density is FactorDensity.DENSE

    def emits_for(
        self,
        _scope: FactorScope,
        _density: FactorDensity,
    ) -> StatCode:
        return StatCode.P_DK
