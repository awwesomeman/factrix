"""Statistical tooling shared across the library.

``Estimator`` — inference-method protocol selected via family-verb
``estimator=`` kwarg (#170).
``NeweyWest`` — reference ``Estimator`` for the procedure-canonical
NW HAC inference path.
``HansenHodrick`` — rectangular-kernel HAC variant for IC / FM PANEL
on overlapping forward returns.
``multiple_testing`` — BHY procedure for FDR control across many factors.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from factrix.stats._estimator import Estimator
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.hansen_hodrick import HansenHodrick
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
from factrix.stats.newey_west import NeweyWest

# Internal registry consumed by `factrix.list_estimators`. Append new
# Estimator instances here as they land — `list_estimators` filters by
# `applicable_to(scope, signal)` so the registry is the single source
# of truth for "which estimators exist".
_ESTIMATOR_REGISTRY: tuple[Estimator, ...] = (NeweyWest(), HansenHodrick())

__all__ = [
    "Estimator",
    "HansenHodrick",
    "NeweyWest",
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "stationary_bootstrap_resamples",
]
