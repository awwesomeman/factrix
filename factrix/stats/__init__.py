"""Statistical tooling shared across the library.

``Estimator`` — base inference-method identity protocol selected via
the slice-test ``estimator=`` kwarg; pure identity semantics
(``name`` / ``description``, no ``compute``). Series-mean HAC inference
lives in ``factrix.inference``.
``InferenceResult`` — harmonized return shape (canonical home is
``factrix.inference``; re-exported here).
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
    InferenceResult,
)
from factrix.stats.block_bootstrap import BlockBootstrap
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.driscoll_kraay import DriscollKraay
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
from factrix.stats.wald_cluster import WaldNWCluster, WaldTwoWayCluster

__all__ = [
    "BlockBootstrap",
    "DriscollKraay",
    "Estimator",
    "InferenceResult",
    "WaldNWCluster",
    "WaldTwoWayCluster",
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "stationary_bootstrap_resamples",
]
