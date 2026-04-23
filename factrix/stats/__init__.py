"""Statistical tooling shared across the library.

``multiple_testing`` — BHY procedure for FDR control across many factors.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.multiple_testing import bhy_adjust, bhy_adjusted_p

__all__ = [
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "stationary_bootstrap_resamples",
]
