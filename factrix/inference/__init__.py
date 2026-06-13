"""``factrix.inference`` — curated statistical inference methods.

A small, curated set of named inference methods, each a frozen dataclass
that carries its own ``compute`` plus identity labels. Discovery is by
the ``fx.inference.*`` namespace (autocomplete + docs); there is no
flat global ``list()`` — the contextual question "which methods does this
metric accept?" is answered per-metric (e.g. ``ic`` accepts
``NON_OVERLAPPING`` / ``NEWEY_WEST``).

This release scopes the namespace to the **series-mean** family
(``compute(df, *, value_col, forward_periods)``). Slice / panel methods
keep their multivariate compute in ``factrix.slicing`` until they move
onto the same ``metric(inference=...)`` path; they are deliberately not
listed here to avoid re-creating a heterogeneous discovery surface.
"""

from __future__ import annotations

from factrix.inference._base import Inference, InferenceResult
from factrix.inference.series_mean import (
    HANSEN_HODRICK,
    NEWEY_WEST,
    NON_OVERLAPPING,
    HansenHodrick,
    NeweyWest,
    NonOverlapping,
)

__all__ = [
    "HANSEN_HODRICK",
    "NEWEY_WEST",
    "NON_OVERLAPPING",
    "HansenHodrick",
    "Inference",
    "InferenceResult",
    "NeweyWest",
    "NonOverlapping",
]
