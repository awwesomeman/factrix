"""Statistical tooling shared across the library.

``Estimator`` — base inference-method protocol selected via family-verb
``estimator=`` kwarg (#170); pure selection semantics (no ``compute``).
``HACEstimator(Estimator)`` — sub-protocol adding cell-internal
``compute(series, *, forward_periods) -> InferenceResult`` for HAC-on-
mean inference (#163). ``NeweyWest`` / ``HansenHodrick`` implement it.
``InferenceResult`` — harmonized return shape for ``HACEstimator.compute``.
``NeweyWest`` — Newey-West HAC ``HACEstimator``; default for
``AnalysisConfig.estimator``.
``HansenHodrick`` — rectangular-kernel HAC variant for IC / FM PANEL
on overlapping forward returns.
``WaldNWCluster`` / ``WaldTwoWayCluster`` — cluster-robust Wald χ²
Estimators for slice contrasts (#153); remain on base ``Estimator``.
``BlockBootstrap`` — block-bootstrap empirical-p Estimator for
paired-diff slice tests (#153); remains on base ``Estimator``.
``multiple_testing`` — BHY procedure for FDR control across many factors.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from factrix.stats._estimator import Estimator, HACEstimator, InferenceResult
from factrix.stats.block_bootstrap import BlockBootstrap
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.hansen_hodrick import HansenHodrick
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
from factrix.stats.newey_west import NeweyWest
from factrix.stats.wald_cluster import WaldNWCluster, WaldTwoWayCluster

# Internal registry consumed by `factrix.list_estimators`. Append new
# Estimator instances here as they land — `list_estimators` filters by
# `applicable_to(scope, signal)` so the registry is the single source
# of truth for "which estimators exist". Slice-test Estimators (#153)
# enter the registry with default-constructed instances; callers
# override the defaults by passing an explicitly-constructed instance
# to the slice-test verb (#176).
_ESTIMATOR_REGISTRY: tuple[Estimator, ...] = (
    NeweyWest(),
    HansenHodrick(),
    WaldNWCluster(),
    WaldTwoWayCluster(),
    BlockBootstrap(),
)

__all__ = [
    "BlockBootstrap",
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
