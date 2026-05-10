"""Statistical tooling shared across the library.

``Estimator`` — inference-method protocol selected via family-verb
``estimator=`` kwarg (#170).
``NeweyWest`` — reference ``Estimator`` for the procedure-canonical
NW HAC inference path.
``multiple_testing`` — BHY procedure for FDR control across many factors.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from factrix.stats._estimator import Estimator
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p
from factrix.stats.newey_west import NeweyWest

__all__ = [
    "Estimator",
    "NeweyWest",
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "stationary_bootstrap_resamples",
]
