"""Cluster-robust Wald χ² Estimator instances.

Two selection-only instances targeting the slice-test setting
(``slice_pairwise_test`` / ``slice_joint_test``):

- ``WaldNWCluster`` — Newey-West (NW) heteroskedasticity-and-autocorrelation-consistent (HAC) + 1-way cluster on the slice grouping;
  consumes the stacked per-date metric panel.
- ``WaldTwoWayCluster`` — [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011] two-way
  cluster on (date, asset); consumes the raw asset-date panel.
  Reserved interface; no function consumes it until
  ``factor_decomposition`` lands later.

Numerical implementations live in ``factrix._stats.wald``; this
module names the inference path the family functions / slice-test
functions dispatch to.
"""

from __future__ import annotations


class WaldNWCluster:
    """Cluster-robust Wald χ² with Newey-West (NW) Bartlett heteroskedasticity-and-autocorrelation-consistent (HAC) + 1-way slice cluster.

    Backs the slice test on a per-date metric panel: K parallel
    per-slice metric series (information coefficient (IC), FM λ, etc.) are stacked and a Wald
    contrast tests the equality of slice means under joint NW HAC of
    the K-vector. Numerics live in
    ``factrix._stats.wald._wald_nw_cluster_means``.

    Applicability is restricted to ``(INDIVIDUAL, DENSE)`` — the
    PANEL inference cells whose per-date scalars feed the stacked
    panel. ``COMMON`` cells produce one number per date by definition
    of the scope and have no within-cell cross-section to slice over.

    Pass an instance to a slice-test function to make the inference
    choice explicit::

        fx.slice_pairwise_test(panel, ic, by="sector",
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


class WaldTwoWayCluster:
    """Two-way cluster Wald χ² on (date, asset) — [Cameron-Gelbach-Miller (2011)][cameron-gelbach-miller-2011].

    Backs the raw asset-date panel inference path (factor × slice
    interaction with full panel SE). Numerics live in
    ``factrix._stats.wald._wald_two_way_cluster``.

    Reserved interface; no function consumes it until
    ``factor_decomposition`` lands later.
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
