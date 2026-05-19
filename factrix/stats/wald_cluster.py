"""Cluster-robust Wald χ² Estimator instances (#153).

Two ``Estimator`` implementations targeting the slice-test setting
(#176 functions ``slice_pairwise_test`` / ``slice_joint_test``):

- ``WaldNWCluster`` — Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) + 1-way cluster on the slice grouping;
  consumes the stacked per-date metric panel. Emits
  ``StatCode.P_WALD_NWCL`` (paired with ``WALD_NWCL`` on the test
  statistic side).
- ``WaldTwoWayCluster`` — [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011] two-way
  cluster on (date, asset); consumes the raw asset-date panel.
  Emits ``StatCode.P_WALD_TWOWAY``. Interface ships this issue but
  no function consumes it until ``factor_decomposition`` lands
  later; calling ``bhy(estimator=WaldTwoWayCluster())`` against a
  profile produced by ``evaluate()`` lands on a missing-stat error
  until the function populates
  ``profile.stats[StatCode.P_WALD_TWOWAY]``.

Numerical implementations live in ``factrix._stats.wald``; this
module names the inference path the family functions / slice-test
functions dispatch to.
"""

from __future__ import annotations

from factrix._axis import FactorScope, FactorSignal
from factrix._codes import StatCode


class WaldNWCluster:
    """Cluster-robust Wald χ² with Newey-West (NW) Bartlett heteroskedasticity-and-autocorrelation-consistent (HAC) + 1-way slice cluster.

    Backs the slice test on a per-date metric panel: K parallel
    per-slice metric series (information coefficient (IC), FM λ, etc.) are stacked and a Wald
    contrast tests the equality of slice means under joint NW HAC of
    the K-vector. Numerics live in
    ``factrix._stats.wald._wald_nw_cluster_means``.

    Applicability is restricted to ``(INDIVIDUAL, CONTINUOUS)`` — the
    PANEL inference cells whose per-date scalars feed the stacked
    panel. ``COMMON`` cells produce one number per date by definition
    of the scope and have no within-cell cross-section to slice over.

    Pass an instance to a slice-test function to make the inference
    choice explicit::

        fx.slice_pairwise_test(metric=ic, df=panel, label="sector",
                                estimator=WaldNWCluster())

    Constructor takes no arguments in this release; the Bartlett-kernel
    bandwidth is resolved automatically by ``_wald_nw_cluster_means``
    (``floor(T^(1/3))`` default). Explicit-lag knob is a future
    enhancement and would arrive as a keyword-only ``__init__``
    parameter without changing callers using the default.
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return (
            "Cluster-robust Wald χ² (NW Bartlett HAC + 1-way cluster on the "
            "slice grouping) on a stacked per-date metric panel."
        )

    def applicable_to(self, scope: FactorScope, signal: FactorSignal) -> bool:
        return scope is FactorScope.INDIVIDUAL and signal is FactorSignal.CONTINUOUS

    def emits_for(
        self,
        _scope: FactorScope,
        _signal: FactorSignal,
    ) -> StatCode:
        return StatCode.P_WALD_NWCL


class WaldTwoWayCluster:
    """Two-way cluster Wald χ² on (date, asset) — [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011].

    Backs the raw asset-date panel inference path (factor × slice
    interaction with full panel SE). Numerics live in
    ``factrix._stats.wald._wald_two_way_cluster``.

    Interface ships this PR; no function consumes it until
    ``factor_decomposition`` lands later. ``list_estimators`` surfaces
    it for ``(INDIVIDUAL, CONTINUOUS)`` cells — calling
    ``bhy(estimator=WaldTwoWayCluster())`` against a profile produced
    by ``evaluate()`` lands on a missing-stat error pointing at the
    precondition (same pattern as ``HansenHodrick`` on a non-overlapping
    profile).
    """

    @property
    def name(self) -> str:
        return type(self).__name__

    @property
    def description(self) -> str:
        return (
            "Two-way cluster Wald χ² on (date, asset), Cameron-Gelbach-Miller "
            "(2011). Reserved interface for the raw asset-date panel path."
        )

    def applicable_to(self, scope: FactorScope, signal: FactorSignal) -> bool:
        return scope is FactorScope.INDIVIDUAL and signal is FactorSignal.CONTINUOUS

    def emits_for(
        self,
        _scope: FactorScope,
        _signal: FactorSignal,
    ) -> StatCode:
        return StatCode.P_WALD_TWOWAY
