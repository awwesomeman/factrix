"""Statistical tooling shared across the library.

``Estimator`` — base inference-method protocol selected via family-verb
``estimator=`` kwarg (#170); pure selection semantics (no ``compute``).
``HACEstimator(Estimator)`` — sub-protocol adding cell-internal
``compute(series, *, forward_periods) -> InferenceResult`` for HAC-on-
mean inference (#163). ``NeweyWest`` / ``HansenHodrick`` implement it.
``InferenceResult`` — harmonized return shape for ``HACEstimator.compute``.
``MomentEstimator(Estimator)`` — sub-protocol adding ``compute(moments,
*, forward_periods) -> GMMResult`` for over-identifying-restriction
tests on a moment-condition system (#191). ``GMM`` implements it.
``GMMResult`` — harmonized return shape for ``MomentEstimator.compute``.
``GMM`` — Hansen (1982) two-step efficient J-test ``MomentEstimator``;
opt-in via ``AnalysisConfig.moment_estimator``.
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

from factrix.stats._estimator import (
    Estimator,
    GMMResult,
    HACEstimator,
    InferenceResult,
    MomentEstimator,
)
from factrix.stats.block_bootstrap import BlockBootstrap
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.gmm import GMM
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
    GMM(),
    WaldNWCluster(),
    WaldTwoWayCluster(),
    BlockBootstrap(),
)


def get_estimator(name: str) -> Estimator:
    """Look up a registered ``Estimator`` instance by ``name`` (#163).

    Used by ``AnalysisConfig.from_dict`` to rehydrate the ``estimator``
    field from the serialized name string. Returns the registry's
    canonical zero-arg instance; mutate-at-your-own-risk callers should
    construct a fresh instance via the class directly. Returns the
    base ``Estimator`` type — callers needing ``HACEstimator`` semantics
    (e.g. ``AnalysisConfig``) ``isinstance``-narrow at the boundary.

    Raises:
        UnknownEstimatorError: ``name`` is not in the registry; the
            message lists every available estimator name.
    """
    from factrix._errors import UnknownEstimatorError

    for est in _ESTIMATOR_REGISTRY:
        if est.name == name:
            return est
    available = ", ".join(sorted(e.name for e in _ESTIMATOR_REGISTRY))
    raise UnknownEstimatorError(f"unknown estimator {name!r}. Available: {available}")


__all__ = [
    "GMM",
    "BlockBootstrap",
    "Estimator",
    "GMMResult",
    "HACEstimator",
    "HansenHodrick",
    "InferenceResult",
    "MomentEstimator",
    "NeweyWest",
    "WaldNWCluster",
    "WaldTwoWayCluster",
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "get_estimator",
    "stationary_bootstrap_resamples",
]
