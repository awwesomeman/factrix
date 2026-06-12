"""Statistical tooling shared across the library.

``Estimator`` — base inference-method protocol selected via the
family-function ``estimator=`` kwarg; pure selection
semantics (no ``compute``).
``HACEstimator(Estimator)`` — sub-protocol adding cell-internal
``compute(series, *, forward_periods) -> InferenceResult`` for HAC-on-
mean inference. ``NeweyWest`` / ``HansenHodrick`` implement it.
``InferenceResult`` — harmonized return shape for ``HACEstimator.compute``.
``NeweyWest`` — Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) ``HACEstimator``.
``HansenHodrick`` — rectangular-kernel HAC variant for information coefficient (IC) / FM PANEL
on overlapping forward returns.
``WaldNWCluster`` / ``WaldTwoWayCluster`` — cluster-robust Wald χ²
Estimators for slice contrasts; remain on base ``Estimator``.
``DriscollKraay`` — [Driscoll-Kraay (1998)][driscoll-kraay-1998]
cross-section-robust HAC SE for pooled-panel slopes; selection-
only base ``Estimator``, numerics consumed by ``pooled_beta``.
``BlockBootstrap`` — block-bootstrap empirical-p Estimator for
paired-diff slice tests; remains on base ``Estimator``.
``multiple_testing`` — Benjamini-Hochberg-Yekutieli (BHY) procedure for false discovery rate (FDR) control across many factors.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from __future__ import annotations

from factrix.stats._estimator import (
    Estimator,
    HACEstimator,
    InferenceResult,
)
from factrix.stats.block_bootstrap import BlockBootstrap
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.driscoll_kraay import DriscollKraay
from factrix.stats.hansen_hodrick import HansenHodrick
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
from factrix.stats.newey_west import NeweyWest
from factrix.stats.wald_cluster import WaldNWCluster, WaldTwoWayCluster

__all__ = [
    "BlockBootstrap",
    "DriscollKraay",
    "Estimator",
    "HACEstimator",
    "HansenHodrick",
    "InferenceResult",
    "NeweyWest",
    "WaldNWCluster",
    "WaldTwoWayCluster",
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "stationary_bootstrap_resamples",
]
