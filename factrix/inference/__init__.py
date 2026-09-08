"""Curated inference methods for tests on a per-period series mean.

``NON_OVERLAPPING`` strides the series, ``NEWEY_WEST`` applies a Bartlett-
kernel HAC variance to the full series, and ``STATIONARY_BOOTSTRAP`` resamples
blocks from the full series. Each member exposes the same ``compute`` contract
and returns an :class:`InferenceResult` containing the tested alternative,
p-value, estimate, sample size, diagnostics, and warnings.

The ``inference=`` parameter is intentionally a closed, per-metric allowlist.
It is available on ``ic``, ``quantile_spread``, ``quantile_spread_vw``, and
``k_spread``; other metrics use an estimator fixed by their statistical shape.
Passing a method outside a metric's allowlist raises
:class:`~factrix.IncompatibleInferenceError` rather than falling back.

``HansenHodrick`` is available only from ``factrix.inference.series_mean`` for
standalone comparisons. Its rectangular kernel has no positive-semidefinite
guarantee, so no registered metric admits it.

See [Inference selection](../development/architecture.md#inference-selection)
for why the union is closed and what the per-metric allowlist records,
[Statistical methods](../reference/statistical-methods.md#nw-hac) for selection
guidance, and the
[calibration reference](../reference/inference-calibration.md) for measured
regimes and known limits.
"""

from __future__ import annotations

from factrix.inference._base import Inference, InferenceResult
from factrix.inference.series_mean import (
    NEWEY_WEST,
    NON_OVERLAPPING,
    STATIONARY_BOOTSTRAP,
    NeweyWest,
    NonOverlapping,
    StationaryBootstrap,
)

__all__ = [
    "NEWEY_WEST",
    "NON_OVERLAPPING",
    "STATIONARY_BOOTSTRAP",
    "Inference",
    "InferenceResult",
    "NeweyWest",
    "NonOverlapping",
    "StationaryBootstrap",
]
