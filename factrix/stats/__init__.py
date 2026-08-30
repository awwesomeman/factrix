"""Statistical tooling shared across the library.

The slice-test selection classes carry pure identity semantics
(``name`` / ``description``, no ``compute``); their numerics live in
the slice-test dispatch path and the ``factrix._stats`` kernels.
Series-mean HAC inference lives in ``factrix.inference``.
``WaldNWCluster`` / ``WaldTwoWayCluster`` — cluster-robust Wald χ²
selection-only instances for slice contrasts.
``DriscollKraay`` — [Driscoll-Kraay (1998)][driscoll-kraay-1998]
cross-section-robust HAC SE for pooled-panel slopes; selection-only
identity handle, numerics consumed by ``pooled_beta``.
``multiple_testing`` — BHY false-discovery plus Holm and Romano-Wolf
family-wise error-rate adjustments across a declared search family.
``bootstrap`` — stationary-bootstrap resampling + CI for dependent series.
"""

from __future__ import annotations

from factrix.stats.bootstrap import (
    bootstrap_mean_ci,
    stationary_bootstrap_resamples,
)
from factrix.stats.driscoll_kraay import DriscollKraay
from factrix.stats.multiple_testing import (
    bhy_adjust,
    bhy_adjusted_p,
    holm_adjusted_p,
    romano_wolf_adjusted_p,
)
from factrix.stats.wald_cluster import WaldNWCluster, WaldTwoWayCluster

__all__ = [
    "DriscollKraay",
    "WaldNWCluster",
    "WaldTwoWayCluster",
    "bhy_adjust",
    "bhy_adjusted_p",
    "bootstrap_mean_ci",
    "holm_adjusted_p",
    "romano_wolf_adjusted_p",
    "stationary_bootstrap_resamples",
]
