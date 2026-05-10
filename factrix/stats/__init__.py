"""Statistical tooling shared across the library.

``Estimator`` — inference-method protocol selected via family-verb
``estimator=`` kwarg (#170).
``multiple_testing`` — BHY procedure for FDR control across many factors.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from factrix.stats._estimator import Estimator
from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p

__all__ = [
    "Estimator",
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "stationary_bootstrap_resamples",
]
