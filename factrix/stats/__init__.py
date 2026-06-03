"""Statistical tooling shared across the library.

``Estimator`` — base inference-method protocol selected via the
family-function ``estimator=`` kwarg (#170); pure selection
semantics (no ``compute``).
``HACEstimator(Estimator)`` — sub-protocol adding cell-internal
``compute(series, *, forward_periods) -> InferenceResult`` for HAC-on-
mean inference (#163). ``NeweyWest`` / ``HansenHodrick`` implement it.
``InferenceResult`` — harmonized return shape for ``HACEstimator.compute``.
``MomentEstimator(Estimator)`` — sub-protocol adding ``compute(moments,
*, forward_periods) -> GMMResult`` for over-identifying-restriction
tests on a moment-condition system (#191). ``GMM`` implements it.
``GMMResult`` — harmonized return shape for ``MomentEstimator.compute``.
``GMM`` — [Hansen (1982)][hansen-1982] two-step efficient J-test ``MomentEstimator``;
opt-in via ``AnalysisConfig.moment_estimator``.
``NeweyWest`` — Newey-West heteroskedasticity-and-autocorrelation-consistent (HAC) ``HACEstimator``; default for
``AnalysisConfig.estimator``.
``HansenHodrick`` — rectangular-kernel HAC variant for information coefficient (IC) / FM PANEL
on overlapping forward returns.
``WaldNWCluster`` / ``WaldTwoWayCluster`` — cluster-robust Wald χ²
Estimators for slice contrasts (#153); remain on base ``Estimator``.
``BlockBootstrap`` — block-bootstrap empirical-p Estimator for
paired-diff slice tests (#153); remains on base ``Estimator``.
``multiple_testing`` — Benjamini-Hochberg-Yekutieli (BHY) procedure for false discovery rate (FDR) control across many factors.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from __future__ import annotations

from typing import Any, Literal

from factrix._axis import FactorScope, FactorDensity
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
    bootstrap_mean_ci_batch,
    stationary_bootstrap_resamples,
)
from factrix.stats.gmm import GMM
from factrix.stats.hansen_hodrick import HansenHodrick
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
from factrix.stats.newey_west import NeweyWest
from factrix.stats.wald_cluster import WaldNWCluster, WaldTwoWayCluster

# Internal registry consumed by `factrix.list_estimators`. Append new
# Estimator instances here as they land — `list_estimators` filters by
# `applicable_to(scope, density)` so the registry is the single source
# of truth for "which estimators exist". Slice-test Estimators (#153)
# enter the registry with default-constructed instances; callers
# override the defaults by passing an explicitly-constructed instance
# to the slice-test function (#176).
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


def list_estimators(
    scope: FactorScope,
    density: FactorDensity,
    *,
    format: Literal["text", "json"] = "text",
    with_import: bool = False,
) -> list[str] | list[dict[str, Any]]:
    """Return Estimator instances applicable to ``(scope, density)``.

    Mirrors :func:`factrix.list_metrics` shape so callers can build a
    single pre-flight pattern: ``list_metrics`` says which scalars a
    cell can emit, ``list_estimators`` says which inference methods
    can drive the family function's ``estimator=`` kwarg for that
    cell. DataStructure is intentionally not an input — Estimator applicability
    is not structure-dependent.

    Args:
        scope: Cell axis to filter on (``FactorScope.INDIVIDUAL`` or
            ``FactorScope.COMMON``).
        density: Cell axis to filter on (``FactorDensity.DENSE`` or
            ``FactorDensity.SPARSE``).
        format: ``"text"`` (default) returns Estimator names sorted
            alphabetically. ``"json"`` returns ``list[dict]`` rows with
            keys ``name``, ``description``, ``import_path``.
        with_import: ``"text"`` only. When ``True``, returns
            ``"name → import_path"`` two-column lines so each row is
            copy-paste-ready into ``from factrix.stats import <name>``.
            Ignored under JSON (``import_path`` is always present
            there).

    Raises:
        IncompatibleAxisError: ``(scope, density)`` matches no
            registered Estimator. ``NeweyWest`` applies to every
            user-facing cell, so as long as it stays in the registry
            this branch is defensive.

    Examples:
        Discover applicable estimators for an INDIVIDUAL × DENSE cell:

        >>> import factrix as fx
        >>> names = fx.list_estimators(
        ...     fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE,
        ... )

        JSON form for tooling:

        >>> rows = fx.list_estimators(
        ...     fx.FactorScope.INDIVIDUAL, fx.FactorDensity.DENSE, format="json",
        ... )
    """
    from factrix._errors import IncompatibleAxisError

    matches = [e for e in _ESTIMATOR_REGISTRY if e.applicable_to(scope, density)]
    if not matches:
        raise IncompatibleAxisError(
            f"no estimators applicable to (scope={scope.value}, density={density.value})"
        )

    matches.sort(key=lambda e: e.name)

    rows = [
        {
            "name": e.name,
            "description": e.description,
            "import_path": f"factrix.stats.{type(e).__name__}",
        }
        for e in matches
    ]
    if format == "json":
        return rows
    if with_import:
        width = max(len(r["name"]) for r in rows)
        return [f"{r['name']:<{width}} → {r['import_path']}" for r in rows]
    return [r["name"] for r in rows]


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
    "bootstrap_mean_ci_batch",
    "get_estimator",
    "list_estimators",
    "stationary_bootstrap_resamples",
]
