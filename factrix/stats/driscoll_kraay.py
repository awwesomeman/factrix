"""``DriscollKraay`` Estimator — cross-section-robust HAC SE for pooled-panel slopes.

Names the [Driscoll & Kraay (1998)][driscoll-kraay-1998] cross-sectional
dependence-robust standard-error path. Selection-only identity handle
(like ``WaldNWCluster`` / ``WaldTwoWayCluster``): the numerics live in
``factrix._stats.hac._driscoll_kraay_cov`` and are driven from inside
``factrix.metrics.fm_beta.pooled_beta`` — DK consumes the full
``(date, asset)`` panel, not a mean.

``pooled_beta`` reports its t-stat / p-value inside its own
``MetricResult.metadata``, so — like ``WaldTwoWayCluster`` — this is a
reserved identity handle for the slice-test dispatch path.
"""

from __future__ import annotations


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
    them directly.

    Pass an instance to a function to make the inference choice explicit.
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
